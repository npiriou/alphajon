import hashlib

import numpy as np

from .break_features import extract_break_observation, observation_size as break_observation_size

LEGACY_ITEM_BUCKETS = 64
ITEM_CLASS_BUCKETS = 320
HOOK_BUCKETS = 8
ITEM_EXTRA_FEATURES = 19
V3_ITEM_BUILD_FEATURES = ITEM_CLASS_BUCKETS * 5 + 8
ITEM_MECHANIC_FEATURES = 49
ITEM_BUILD_FEATURES = V3_ITEM_BUILD_FEATURES + ITEM_MECHANIC_FEATURES
LEGACY_ITEM_EXTRA_FEATURES = 15
FEATURE_VERSION = "item_activation_v4"
V3_FEATURE_VERSION = "item_activation_v3"
V2_FEATURE_VERSION = "item_activation_v2"
LEGACY_FEATURE_VERSION = "item_activation_legacy"
HOOK_ORDER = {
    "en_combat": 0,
    "en_survie": 1,
    "en_fuite": 2,
    "debut_tour": 3,
    "fin_tour": 4,
    "en_vaincu": 5,
    "en_subit_dommages": 6,
}

SCRY_ITEM_CLASSES = {
    "PommeDAdam",
    "BouleDeCristal",
    "BinoclesDeLInventeur",
    "CleDeSalomon",
    "CompasDuCapitaine",
    "FilDuDestin",
    "JournalDuFutur",
    "MiroirDeYata",
    "OeilDHorus",
    "OiseauDeMauvaisAugure",
}

PEEK_COMBO_ITEM_CLASSES = {
    "CapeDInvisibilite",
    "GriffesEclair",
    "HacheMystique",
    "PistoletLaser",
}

DECK_MANIPULATION_ITEM_CLASSES = {
    "AnneauDuVent",
    "BananeExperimentale",
    "BottesDePoncage",
    "CapeDInvisibilite",
    "ClocheDuDejaVu",
    "CouteauxDeLancer",
    "FilDuDestin",
    "HacheMystique",
    "LanterneAbsorbante",
    "OeilDHorus",
    "OiseauDeMauvaisAugure",
    "PierreDePressentiment",
    "PommeDAdam",
    "TambourDeKui",
    "VoileDIsis",
}

REPAIR_ITEM_CLASSES = {
    "BouclierDragon",
    "CleAmulette",
    "CoffreAnime",
    "CouteauSuisse",
    "GantsDeGaia",
    "Paratonnerre",
    "PierreDePressentiment",
    "PotionFeerique",
    "YoYoProtecteur",
}

STEAL_OR_DENIAL_ITEM_CLASSES = {
    "CarapaceBleue",
    "CorneDAbordage",
    "CraneDuRoiLiche",
    "DagueDeBrutus",
    "EnclumeInstable",
    "EspritDuDonjon",
    "EventailMaudit",
    "FouetDuFourbe",
    "ParfumRegenerant",
    "SiegeDeTroie",
}

REPLAY_OR_TEMPO_ITEM_CLASSES = {
    "BotteDePandore",
    "BoomerangMystique",
    "ChapeauDuNovice",
    "CouteauQuiTombe",
    "GriffesEclair",
    "PierreDePressentiment",
    "PlanPresqueParfait",
    "TronconneuseEnflammee",
}

SCRY_HERO_CLASSES = {
    "Prophete",
    "LapinBlanc",
}


def _item_class_index(item):
    from objets import objets_disponibles

    cls = type(item)
    seen = []
    for obj in objets_disponibles:
        obj_cls = type(obj)
        if obj_cls not in seen:
            seen.append(obj_cls)
    try:
        return seen.index(cls)
    except ValueError:
        return -1


def _set_class_feature(features, item, value):
    class_index = _item_class_index(item)
    if 0 <= class_index < ITEM_CLASS_BUCKETS:
        features[class_index] += float(value)
    else:
        features[_bucket(getattr(item, "nom", type(item).__name__), ITEM_CLASS_BUCKETS)] += float(value)


def _bucket(text, buckets):
    raw = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(raw[:4], "little") % buckets


def _common_item_features(joueur, item, card, hook):
    item_index = joueur.objets.index(item) if item in joueur.objets else -1
    card_power = getattr(card, "puissance", getattr(card, "puissance_initiale", 0))
    card_damage = getattr(card, "dommages", card_power)
    return item_index, card_power, card_damage, [
        item_index / 12.0,
        1.0 if getattr(item, "intact", False) else 0.0,
        1.0 if getattr(item, "actif", False) else 0.0,
        getattr(item, "pv_bonus", 0) / 10.0,
        getattr(item, "modificateur_de", 0) / 10.0,
        getattr(item, "priorite", 0) / 100.0,
        len(getattr(item, "types_tags", ())) / 8.0,
        len(getattr(item, "puissance_tags", ())) / 10.0,
        1.0 if getattr(item, "pv_bonus", 0) >= joueur.pv_total else 0.0,
        card_power / 10.0,
        card_damage / 10.0,
        1.0 if card_damage >= joueur.pv_total else 0.0,
        1.0 if getattr(card, "event", False) else 0.0,
        1.0 if getattr(card, "is_X", False) else 0.0,
        1.0 if hook == "en_combat" else 0.0,
    ]


def _remaining_cards(jeu):
    donjon = jeu.donjon
    return [donjon.cartes[i] for i in donjon.ordre[donjon.index:]]


def _known_window_features(joueur, jeu, max_cards=4):
    cards = _remaining_cards(jeu)
    known_cards = getattr(joueur, "cartes_connues", set())
    window = cards[:max_cards]
    features = []
    known_count = 0
    known_lethal = 0
    known_easy = 0
    known_event = 0
    known_danger_sum = 0.0
    known_best = 0.0
    known_worst = 0.0

    for card in window:
        known = card in known_cards
        is_event = bool(getattr(card, "event", False))
        power = 0.0 if is_event else float(getattr(card, "puissance_initiale", getattr(card, "puissance", 0)))
        lethal = bool((not is_event) and joueur._degats_attendus(card, jeu) >= joueur.pv_total)
        easy = bool((not is_event) and joueur.peut_executer_facilement(card))
        danger = 0.0
        if known:
            known_count += 1
            if is_event:
                known_event += 1
                danger = -1.0
            else:
                known_lethal += int(lethal)
                known_easy += int(easy)
                danger = power / 10.0
                if lethal:
                    danger += 1.0
                if easy:
                    danger -= 0.5
            known_danger_sum += danger
            known_best = min(known_best, danger)
            known_worst = max(known_worst, danger)
        features.extend([
            1.0 if known else 0.0,
            1.0 if known and is_event else 0.0,
            power / 10.0 if known and not is_event else 0.0,
            1.0 if known and lethal else 0.0,
            1.0 if known and easy else 0.0,
        ])

    missing = max_cards - len(window)
    if missing > 0:
        features.extend([0.0] * missing * 5)

    denom = max(1, len(window))
    known_denom = max(1, known_count)
    features.extend([
        known_count / float(max_cards),
        known_lethal / float(known_denom),
        known_easy / float(known_denom),
        known_event / float(known_denom),
        known_danger_sum / float(known_denom),
        known_best,
        known_worst,
        1.0 if cards and cards[0] in known_cards else 0.0,
        len(cards) / 60.0,
        len(window) / float(max_cards),
        sum(1 for c in cards[:8] if c in known_cards) / 8.0,
        denom / float(max_cards),
    ])
    return features


def _mechanic_features(joueur, jeu, item, hook):
    cls_name = type(item).__name__
    hero_cls = type(getattr(joueur, "perso_obj", None)).__name__
    hero_level = int(getattr(getattr(joueur, "perso_obj", None), "level", 1))
    opponents = [j for j in getattr(jeu, "joueurs", []) if j is not joueur]
    opponent_scores = [j._score_rapide() for j in opponents if getattr(j, "vivant", False)]
    opponent_best = max(opponent_scores, default=0)
    own_score = joueur._score_rapide()
    broken_items = sum(1 for obj in getattr(joueur, "objets", []) if not getattr(obj, "intact", False))
    opponent_pile = sum(len(getattr(j, "pile_monstres_vaincus", [])) for j in opponents)
    opponents_in_dungeon = sum(1 for j in opponents if getattr(j, "dans_le_dj", False))
    own_pile = len(getattr(joueur, "pile_monstres_vaincus", []))
    score_gap = own_score - opponent_best

    return _known_window_features(joueur, jeu) + [
        1.0 if cls_name in SCRY_ITEM_CLASSES else 0.0,
        1.0 if cls_name in PEEK_COMBO_ITEM_CLASSES else 0.0,
        1.0 if cls_name in DECK_MANIPULATION_ITEM_CLASSES else 0.0,
        1.0 if cls_name in REPAIR_ITEM_CLASSES else 0.0,
        1.0 if cls_name in STEAL_OR_DENIAL_ITEM_CLASSES else 0.0,
        1.0 if cls_name in REPLAY_OR_TEMPO_ITEM_CLASSES else 0.0,
        1.0 if hero_cls in SCRY_HERO_CLASSES else 0.0,
        1.0 if hero_cls == "Prophete" else 0.0,
        1.0 if hero_cls == "LapinBlanc" else 0.0,
        hero_level / 2.0,
        1.0 if hook in ("debut_tour", "fin_tour", "en_vaincu", "en_subit_dommages") else 0.0,
        broken_items / 6.0,
        opponent_pile / 30.0,
        opponents_in_dungeon / max(1.0, float(len(opponents))),
        own_pile / 30.0,
        score_gap / 30.0,
        1.0 if score_gap < 0 else 0.0,
    ]


def extract_item_activation_observation_legacy(joueur, jeu, item, card, hook):
    base = extract_break_observation(joueur, jeu).tolist()
    _, _, _, item_features = _common_item_features(joueur, item, card, hook)
    item_one_hot = [0.0] * LEGACY_ITEM_BUCKETS
    item_one_hot[_bucket(getattr(item, "nom", type(item).__name__), LEGACY_ITEM_BUCKETS)] = 1.0
    return np.asarray(base + item_features + item_one_hot, dtype=np.float32)


def _build_context_features(joueur, jeu, item):
    own_intact = [0.0] * ITEM_CLASS_BUCKETS
    own_broken = [0.0] * ITEM_CLASS_BUCKETS
    own_active = [0.0] * ITEM_CLASS_BUCKETS
    opponent_intact = [0.0] * ITEM_CLASS_BUCKETS
    opponent_active = [0.0] * ITEM_CLASS_BUCKETS

    for obj in joueur.objets:
        if getattr(obj, "intact", False):
            _set_class_feature(own_intact, obj, 1.0 / 6.0)
        else:
            _set_class_feature(own_broken, obj, 1.0 / 6.0)
        if getattr(obj, "actif", False):
            _set_class_feature(own_active, obj, 1.0 / 6.0)

    opponents = [j for j in getattr(jeu, "joueurs", []) if j is not joueur]
    opponent_slots = max(1.0, float(sum(len(getattr(j, "objets", [])) for j in opponents)))
    for opponent in opponents:
        for obj in getattr(opponent, "objets", []):
            if getattr(obj, "intact", False):
                _set_class_feature(opponent_intact, obj, 1.0 / opponent_slots)
            if getattr(obj, "actif", False):
                _set_class_feature(opponent_active, obj, 1.0 / opponent_slots)

    same_class_count = sum(1 for obj in joueur.objets if type(obj) is type(item))
    active_count = sum(1 for obj in joueur.objets if getattr(obj, "actif", False))
    broken_count = sum(1 for obj in joueur.objets if not getattr(obj, "intact", False))
    combat_items = sum(1 for obj in joueur.objets if hasattr(obj, "en_combat"))
    survival_items = sum(1 for obj in joueur.objets if hasattr(obj, "en_survie"))
    flee_items = sum(1 for obj in joueur.objets if hasattr(obj, "en_fuite"))
    opponent_best_score = max((j._score_rapide() for j in opponents if j.vivant), default=0)
    own_score = joueur._score_rapide()
    scalars = [
        same_class_count / 6.0,
        active_count / 6.0,
        broken_count / 6.0,
        combat_items / 6.0,
        survival_items / 6.0,
        flee_items / 6.0,
        (own_score - opponent_best_score) / 30.0,
        opponent_best_score / 30.0,
    ]
    return own_intact + own_broken + own_active + opponent_intact + opponent_active + scalars


def extract_item_activation_observation_v2(joueur, jeu, item, card, hook):
    base = extract_break_observation(joueur, jeu).tolist()
    item_index, card_power, card_damage, item_features = _common_item_features(joueur, item, card, hook)
    item_features.extend([
        item_index / max(1.0, float(len(joueur.objets) - 1)),
        joueur.pv_total / 30.0,
        len([obj for obj in joueur.objets if getattr(obj, "intact", False)]) / 12.0,
        1.0 if card_damage > 0 and card_damage < joueur.pv_total <= card_damage + 2 else 0.0,
    ])
    item_one_hot = [0.0] * ITEM_CLASS_BUCKETS
    class_index = _item_class_index(item)
    if 0 <= class_index < ITEM_CLASS_BUCKETS:
        item_one_hot[class_index] = 1.0
    else:
        item_one_hot[_bucket(getattr(item, "nom", type(item).__name__), ITEM_CLASS_BUCKETS)] = 1.0
    hook_one_hot = [0.0] * HOOK_BUCKETS
    hook_one_hot[HOOK_ORDER.get(hook, HOOK_BUCKETS - 1)] = 1.0
    return np.asarray(base + item_features + item_one_hot + hook_one_hot, dtype=np.float32)


def extract_item_activation_observation(joueur, jeu, item, card, hook):
    v2 = extract_item_activation_observation_v2(joueur, jeu, item, card, hook).tolist()
    build_features = _build_context_features(joueur, jeu, item)
    mechanic_features = _mechanic_features(joueur, jeu, item, hook)
    return np.asarray(v2 + build_features + mechanic_features, dtype=np.float32)


def extract_item_activation_observation_v3(joueur, jeu, item, card, hook):
    v2 = extract_item_activation_observation_v2(joueur, jeu, item, card, hook).tolist()
    build_features = _build_context_features(joueur, jeu, item)
    return np.asarray(v2 + build_features, dtype=np.float32)


def legacy_observation_size():
    return break_observation_size() + LEGACY_ITEM_EXTRA_FEATURES + LEGACY_ITEM_BUCKETS


def v2_observation_size():
    return break_observation_size() + ITEM_EXTRA_FEATURES + ITEM_CLASS_BUCKETS + HOOK_BUCKETS


def observation_size():
    return v2_observation_size() + ITEM_BUILD_FEATURES


def v3_observation_size():
    return v2_observation_size() + V3_ITEM_BUILD_FEATURES
