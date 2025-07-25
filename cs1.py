try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'streamlit' module is not installed. Please install it by running 'pip install streamlit' and try again.")

import numpy as np
import random
import pandas as pd

card_types = [
    "1 Star Card", "2 Star Card", "3 Star Card", "4 Star Card", "5 Star Card", "Gold Card"
]
card_counts = {
    "1 Star Card": 17,
    "2 Star Card": 16,
    "3 Star Card": 14,
    "4 Star Card": 13,
    "5 Star Card": 12,
    "Gold Card": 9
}
unique_card_list = []
for ct in card_types:
    unique_card_list += [f"{ct} {i}" for i in range(card_counts[ct])]

set_definitions = [
    {"1 Star Card": 5, "2 Star Card": 4, "3 Star Card": 0, "4 Star Card": 0, "5 Star Card": 0, "Gold Card": 0},  # Set 1
    {"1 Star Card": 4, "2 Star Card": 3, "3 Star Card": 2, "4 Star Card": 0, "5 Star Card": 0, "Gold Card": 0},  # Set 2
    {"1 Star Card": 3, "2 Star Card": 2, "3 Star Card": 2, "4 Star Card": 2, "5 Star Card": 0, "Gold Card": 0},  # Set 3
    {"1 Star Card": 2, "2 Star Card": 2, "3 Star Card": 2, "4 Star Card": 2, "5 Star Card": 1, "Gold Card": 0},  # Set 4
    {"1 Star Card": 1, "2 Star Card": 2, "3 Star Card": 2, "4 Star Card": 2, "5 Star Card": 1, "Gold Card": 1},  # Set 5
    {"1 Star Card": 1, "2 Star Card": 1, "3 Star Card": 2, "4 Star Card": 2, "5 Star Card": 2, "Gold Card": 1},  # Set 6
    {"1 Star Card": 1, "2 Star Card": 1, "3 Star Card": 2, "4 Star Card": 2, "5 Star Card": 2, "Gold Card": 1},  # Set 7
    {"1 Star Card": 0, "2 Star Card": 1, "3 Star Card": 1, "4 Star Card": 2, "5 Star Card": 2, "Gold Card": 3},  # Set 8
    {"1 Star Card": 0, "2 Star Card": 0, "3 Star Card": 1, "4 Star Card": 1, "5 Star Card": 4, "Gold Card": 3},  # Set 9
]
set_names = [f"Set {i+1}" for i in range(9)]

# Default probabilities
pack_data = {
    "M Pack": {"count": 2, "prob": [0.34, 0.27, 0.2, 0.115, 0.065, 0.01]},
    "L Pack": {"count": 3, "prob": [0.31, 0.24, 0.17, 0.16, 0.10, 0.02]},
    "XL Pack": {"count": 4, "prob": [0.27, 0.21, 0.165, 0.18, 0.14, 0.035]},
    "Special Pack": {"count": 6, "prob": [0.235, 0.185, 0.15, 0.195, 0.185, 0.05], "new_guarantee": True},
    "Legendary Pack": {"count": 6, "prob": [0.235, 0.185, 0.15, 0.195, 0.185, 0.05], "new_guarantee": True}
}

st.title("CARD COLLECTION SIMULATION")

st.sidebar.header("Package Order")
st.sidebar.markdown("""
Exp: M Pack:8, L Pack:4, M Pack:4, XL Pack:3, Special Pack:1, XL Pack:1
""")
custom_pack_sequence = st.sidebar.text_input(
    "Package Order and Quantity",
    value="M Pack:3, L Pack:4, XL Pack:3, Special Pack:1, Legendary Pack:1"
)

unique_first_n = st.sidebar.number_input(
    "First N cards are guaranteed unique:",
    min_value=0, max_value=sum(card_counts.values()), value=20, step=1
)

# --- Sidebar: Probabilities dynamic ---
st.sidebar.header("Set Probabilities for Each Pack")

def get_pack_probs_input(pack, default_probs):
    st.sidebar.write(f"### {pack} Probabilities (sum must be 1.00)")
    probs = []
    for i, ct in enumerate(card_types):
        p = st.sidebar.number_input(
            f"{pack} - {ct} Probability",
            min_value=0.0, max_value=1.0, value=float(default_probs[i]), step=0.01, format="%.3f", key=f"{pack}_{ct}"
        )
        probs.append(p)
    prob_sum = sum(probs)
    if abs(prob_sum - 1.0) > 1e-4:
        st.sidebar.error(f"Sum for {pack} = {prob_sum:.3f} (Must be 1.00!)")
    return probs

pack_data_dynamic = {}
for pack in ["M Pack", "L Pack", "XL Pack"]:
    pack_data_dynamic[pack] = {
        "count": pack_data[pack]["count"],
        "prob": get_pack_probs_input(pack, pack_data[pack]["prob"])
    }
for pack in ["Special Pack", "Legendary Pack"]:
    pack_data_dynamic[pack] = dict(pack_data[pack])
    pack_data_dynamic[pack]["prob"] = get_pack_probs_input(pack, pack_data[pack]["prob"])

def parse_custom_pack_sequence(seq_str):
    seq = []
    for part in seq_str.split(","):
        if ":" in part:
            name, count = part.strip().split(":")
            if name.strip() in pack_data_dynamic and count.strip().isdigit():
                seq.extend([name.strip()] * int(count.strip()))
    return seq

def draw_from_pack(
    pack_type, drawn_this_pack, unique_give_count, unique_guarantee, gold_guarantee,
    unique_card_type_sets, pack_data = pack_data_dynamic, special_pack_memory=None, legendary_pack_memory=None,
    set_definitions=None, card_types=None
):
    drawn_cards = []
    all_unique_left = set(unique_card_list)
    for cardset in unique_card_type_sets.values():
        all_unique_left -= cardset
    gold_left = [uc for uc in all_unique_left if uc.startswith("Gold Card")]
    not_drawn_this_pack = lambda ct: [card for card in unique_card_list if card.startswith(ct) and card not in drawn_this_pack]

    # --- Special Pack Yeni Kurallar ---
    if pack_type == "Special Pack":
        # İlk 5 kart: dynamic probability'lerle
        for _ in range(5):
            chosen_type = np.random.choice(card_types, p=pack_data[pack_type]["prob"])
            available = not_drawn_this_pack(chosen_type)
            if available:
                selected = random.choice(available)
                drawn_cards.append(selected)
                drawn_this_pack.add(selected)
        # 6. kart: özel probability
        special_probs_types = ["3 Star Card", "4 Star Card", "5 Star Card", "Gold Card"]
        special_probs = [0.26, 0.34, 0.32, 0.08]
        chosen_type = np.random.choice(special_probs_types, p=special_probs)
        available = not_drawn_this_pack(chosen_type)
        if available:
            selected = random.choice(available)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)

        # unique ve set tamamlama swap işlemleri:
        unique_in_this_pack = [c for c in drawn_cards if c in all_unique_left]
        set_completed_by_one = []
        if set_definitions and card_types:
            do_swap = random.random() < 0.8
            if do_swap and unique_in_this_pack:
                for set_idx, sdef in enumerate(set_definitions):
                    available_uniques = {ct: set(unique_card_type_sets[ct]) | set(unique_in_this_pack) for ct in card_types}
                    need = {ct: sdef[ct] for ct in card_types}
                    eksik = sum([max(0, need[ct] - len(available_uniques[ct])) for ct in card_types])
                    can_complete = all(len(available_uniques[ct]) >= need[ct] for ct in card_types)
                    if eksik == 1 and not can_complete:
                        set_completed_by_one.append((set_idx, need))
                if set_completed_by_one:
                    set_idx, need = set_completed_by_one[0]
                    for ct in card_types:
                        eksik_sayisi = need[ct] - len(unique_card_type_sets[ct])
                        if eksik_sayisi == 1:
                            unique_candidates = [c for c in unique_in_this_pack if c.startswith(ct)]
                            if unique_candidates:
                                idx_in_pack = drawn_cards.index(unique_candidates[0])
                                possible_needed = [c for c in all_unique_left if c.startswith(ct)]
                                if possible_needed:
                                    drawn_cards[idx_in_pack] = possible_needed[0]
            if special_pack_memory is not None:
                unique_fail_streak = 0
                for memory in reversed(special_pack_memory):
                    if not memory:
                        unique_fail_streak += 1
                    else:
                        break
                add_unique = False
                if unique_fail_streak == 1:
                    add_unique = random.random() < 0.33
                elif unique_fail_streak == 2:
                    add_unique = random.random() < 0.66
                elif unique_fail_streak >= 3:
                    add_unique = True
                if add_unique:
                    possible_uniques = list(all_unique_left - set(drawn_this_pack))
                    if possible_uniques:
                        idx_to_replace = random.randint(0, 5)
                        drawn_cards[idx_to_replace] = possible_uniques[0]
                        drawn_this_pack.add(possible_uniques[0])
        return drawn_cards

    # --- Legendary Pack Yeni Kurallar ---
    if pack_type == "Legendary Pack":
        # İlk 5 kart: dynamic probability
        for _ in range(5):
            chosen_type = np.random.choice(card_types, p=pack_data[pack_type]["prob"])
            available = not_drawn_this_pack(chosen_type)
            if available:
                selected = random.choice(available)
                drawn_cards.append(selected)
                drawn_this_pack.add(selected)
        # 6. kart: özel probability
        special_probs_types = ["3 Star Card", "4 Star Card", "5 Star Card", "Gold Card"]
        special_probs = [0.26, 0.34, 0.32, 0.08]
        chosen_type = np.random.choice(special_probs_types, p=special_probs)
        available = not_drawn_this_pack(chosen_type)
        if available:
            selected = random.choice(available)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)
        unique_in_this_pack = [c for c in drawn_cards if c in all_unique_left]
        if set_definitions and card_types:
            do_swap = random.random() < 0.8
            if do_swap and unique_in_this_pack:
                set_completed_by_one = []
                for set_idx, sdef in enumerate(set_definitions):
                    available_uniques = {ct: set(unique_card_type_sets[ct]) | set(unique_in_this_pack) for ct in card_types}
                    need = {ct: sdef[ct] for ct in card_types}
                    eksik = sum([max(0, need[ct] - len(available_uniques[ct])) for ct in card_types])
                    can_complete = all(len(available_uniques[ct]) >= need[ct] for ct in card_types)
                    if eksik == 1 and not can_complete:
                        set_completed_by_one.append((set_idx, need))
                if set_completed_by_one:
                    set_idx, need = set_completed_by_one[0]
                    for ct in card_types:
                        eksik_sayisi = need[ct] - len(unique_card_type_sets[ct])
                        if eksik_sayisi == 1:
                            unique_candidates = [c for c in unique_in_this_pack if c.startswith(ct)]
                            if unique_candidates:
                                idx_in_pack = drawn_cards.index(unique_candidates[0])
                                possible_needed = [c for c in all_unique_left if c.startswith(ct)]
                                if possible_needed:
                                    drawn_cards[idx_in_pack] = possible_needed[0]
        possible_uniques = list(all_unique_left - set(drawn_this_pack))
        non_unique_in_pack = [c for c in drawn_cards if c not in all_unique_left]
        if possible_uniques and non_unique_in_pack:
            idx_to_replace = drawn_cards.index(non_unique_in_pack[0])
            drawn_cards[idx_to_replace] = possible_uniques[0]
            drawn_this_pack.add(possible_uniques[0])
        return drawn_cards

    # --- Diğer paketler için eski mantık ---
    for _ in range(min(pack_data[pack_type]["count"], unique_give_count)):
        possible_new = list(all_unique_left - set(drawn_this_pack))
        if possible_new:
            selected = random.choice(possible_new)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)
            ct = next(ct for ct in card_types if selected.startswith(ct))
            unique_card_type_sets[ct].add(selected)
        else:
            break
    if unique_guarantee and len(drawn_cards) < pack_data[pack_type]["count"]:
        possible_new = list(all_unique_left - set(drawn_this_pack))
        if possible_new:
            selected = random.choice(possible_new)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)
            ct = next(ct for ct in card_types if selected.startswith(ct))
            unique_card_type_sets[ct].add(selected)
    if gold_guarantee and len(drawn_cards) < pack_data[pack_type]["count"]:
        gold_not_drawn = list(set(gold_left) - set(drawn_this_pack))
        if gold_not_drawn:
            selected = random.choice(gold_not_drawn)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)
            ct = next(ct for ct in card_types if selected.startswith(ct))
            unique_card_type_sets[ct].add(selected)
    while len(drawn_cards) < pack_data[pack_type]["count"]:
        total_avail_by_type = [len([c for c in unique_card_list if c.startswith(ct) and c not in drawn_this_pack]) for ct in card_types]
        total_avail = sum(total_avail_by_type)
        if total_avail == 0:
            break
        dynamic_probs = [t / total_avail if total_avail > 0 else 0 for t in total_avail_by_type]
        chosen_type = np.random.choice(card_types, p=dynamic_probs)
        available = not_drawn_this_pack(chosen_type)
        if available:
            selected = random.choice(available)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)
        else:
            continue
    return drawn_cards

def unique_sets_distribution_with_steps(unique_card_type_sets, set_definitions, card_types):
    available_uniques = {ct: set(s) for ct, s in unique_card_type_sets.items()}
    sets_completed = []
    sets_completion_steps = []
    for s in set_definitions:
        can_complete = True
        used_this_set = {}
        for ct in card_types:
            need = s[ct]
            have = len(available_uniques[ct])
            if need > have:
                can_complete = False
            used_this_set[ct] = set(list(available_uniques[ct])[:need])
        if can_complete:
            for ct in card_types:
                available_uniques[ct] -= used_this_set[ct]
            sets_completed.append(True)
            sets_completion_steps.append(None)  # Adım sonradan doldurulacak
        else:
            sets_completed.append(False)
            sets_completion_steps.append(None)
    return sets_completed, sets_completion_steps

def eksik_kart_hesapla_for_set(unique_card_type_sets, set_definitions, card_types, set_idx):
    # Tüm setleri unique kart havuzunu eksilterek sırayla simüle et
    available_uniques = {ct: set(s) for ct, s in unique_card_type_sets.items()}
    for i, s in enumerate(set_definitions):
        need = {ct: s[ct] for ct in card_types}
        used = {ct: set(list(available_uniques[ct])[:need[ct]]) for ct in card_types}
        can_complete = all(len(available_uniques[ct]) >= need[ct] for ct in card_types)
        if i == set_idx:
            if can_complete:
                return 0
            else:
                eksik = 0
                for ct in card_types:
                    eksik += max(0, need[ct] - len(available_uniques[ct]))
                return eksik
        if can_complete:
            for ct in card_types:
                available_uniques[ct] -= used[ct]


def simulate_once(
    pack_sequence, unique_first_n=20, return_cumulative=False,
    special_pack_memory=None, legendary_pack_memory=None, pack_data=pack_data_dynamic
):
    all_history = []
    unique_card_type_sets = {ct: set() for ct in card_types}
    unique_cards_overall = set()
    unique_give_count = unique_first_n
    cumulative_total = {ct: [] for ct in card_types}
    cumulative_unique = {ct: [] for ct in card_types}
    special_pack_memory = special_pack_memory if special_pack_memory is not None else []
    legendary_pack_memory = legendary_pack_memory if legendary_pack_memory is not None else []
    special_counter = 0
    legendary_counter = 0

    for pack_type in pack_sequence:
        drawn_this_pack = set()
        unique_guarantee = pack_data.get(pack_type, {}).get("new_guarantee", False)
        gold_guarantee = pack_data.get(pack_type, {}).get("gold_guarantee", False)
        is_special = (pack_type == "Special Pack")
        is_legendary = (pack_type == "Legendary Pack")

        drawn_cards = draw_from_pack(
            pack_type, drawn_this_pack,
            unique_give_count=unique_give_count,
            unique_guarantee=unique_guarantee,
            gold_guarantee=gold_guarantee,
            unique_card_type_sets=unique_card_type_sets,
            pack_data=pack_data,
            special_pack_memory=special_pack_memory if is_special else None,
            legendary_pack_memory=legendary_pack_memory if is_legendary else None,
            set_definitions=set_definitions, card_types=card_types
        )

        # unique memory update
        if is_special:
            special_counter += 1
            any_unique = any([c not in all_history for c in drawn_cards])
            special_pack_memory.append(any_unique)
        if is_legendary:
            legendary_counter += 1
            any_unique = any([c not in all_history for c in drawn_cards])
            legendary_pack_memory.append(any_unique)

        unique_give_count = max(0, unique_give_count - sum([c not in all_history for c in drawn_cards]))
        for card in drawn_cards:
            ct = next(ct for ct in card_types if card.startswith(ct))
            all_history.append(card)
            unique_cards_overall.add(card)
            unique_card_type_sets[ct].add(card)
            # Cumulative tablo güncellemesi (total ve unique)
            for typ in card_types:
                cumulative_total[typ].append(len([c for c in all_history if c.startswith(typ)]))
                cumulative_unique[typ].append(len(unique_card_type_sets[typ]))
    sets_completed_result, _ = unique_sets_distribution_with_steps(unique_card_type_sets, set_definitions, card_types)
    if return_cumulative:
        return all_history, unique_card_type_sets, sets_completed_result, cumulative_total, cumulative_unique
    else:
        return all_history, unique_card_type_sets, sets_completed_result
      
pack_sequence = parse_custom_pack_sequence(custom_pack_sequence)

if st.sidebar.button("Start Simulation"):

    # Simülasyon önce çalıştırılmalı!
    all_history, unique_card_type_sets, sets_completed_result, cumulative_total, cumulative_unique = simulate_once(
        pack_sequence, unique_first_n=unique_first_n, return_cumulative=True
    )

    # Her karta doğru paketi eşle
    pack_for_each_card = []
    idx = 0
    for pack in pack_sequence:
      for _ in range(pack_data_dynamic[pack]["count"]):
        if idx < len(all_history):
            pack_for_each_card.append(pack)
            idx += 1

    # Eğer kart sayısı fazla ise trimle
    pack_for_each_card = pack_for_each_card[:len(all_history)]

    card_rows = []
    cumulative_unique_now = {ct: set() for ct in card_types}
    completed_sets_by_step = set()
    card_count_so_far = 0
    set_step_list = []
    completed_step_dict = {}

    for idx, card in enumerate(all_history):
        ct = next(ct for ct in card_types if card.startswith(ct))
        cumulative_unique_now[ct].add(card)
        card_count_so_far += 1
        sets_completed, _ = unique_sets_distribution_with_steps(cumulative_unique_now, set_definitions, card_types)
        completed_now = []
        for sidx, completed in enumerate(sets_completed):
            if completed and (sidx not in completed_sets_by_step):
                completed_now.append(set_names[sidx])
                completed_sets_by_step.add(sidx)
                completed_step_dict[set_names[sidx]] = idx + 1
        set_step_list.append(", ".join(completed_now) if completed_now else "")
        card_rows.append({
            "Selection Order": idx + 1,
            "Pack": pack_for_each_card[idx] if idx < len(pack_for_each_card) else "-",
            "Card": card,
            "Card Type": ct,
            "Total Unique Card": sum(len(s) for s in cumulative_unique_now.values()),
            "Total Card": idx + 1,
            "Completed Sets": set_step_list[-1]
        })

    st.subheader("All Selected Cards")
    st.dataframe(pd.DataFrame(card_rows))


    # Cumulative tabloyu oluştururken unique/total'ları yan yana sırala ve set tamamlanma adımı ekle
    cumu_cols = []
    cumu_dict = {}
    for ct in card_types:
        cumu_cols += [f"{ct} Unique", f"{ct} Total"]
        cumu_dict[f"{ct} Unique"] = cumulative_unique[ct]
        cumu_dict[f"{ct} Total"] = cumulative_total[ct]
    steps = list(range(1, len(all_history) + 1))
    # Her adımda tamamlanan ilk setin adımını ekle
    set_done_step_list = []
    completed_step_tracker = set()
    cumulative_unique_now2 = {ct: set() for ct in card_types}
    for idx, card in enumerate(all_history):
        ct = next(ct for ct in card_types if card.startswith(ct))
        cumulative_unique_now2[ct].add(card)
        sets_completed, _ = unique_sets_distribution_with_steps(cumulative_unique_now2, set_definitions, card_types)
        this_step_done = ""
        for sidx, completed in enumerate(sets_completed):
            if completed and (sidx not in completed_step_tracker):
                this_step_done = set_names[sidx]
                completed_step_tracker.add(sidx)
                break
        set_done_step_list.append(this_step_done if this_step_done else "")
    df_cum = pd.DataFrame({**cumu_dict, "Step": steps, "Completed Sets": set_done_step_list})
    # Sütun sıralama: unique-total yan yana ve en sonda step ve set tamamlandı
    ordered_cols = []
    for ct in card_types:
        ordered_cols += [f"{ct} Unique", f"{ct} Total"]
    ordered_cols += ["Step", "Completed Sets"]
    st.subheader("Cumulative Card Selection Table (Unique & Total)")
    st.dataframe(df_cum[ordered_cols])

    # Tamamlanan setler ve adımda tamamlanma tablosu
    set_finish_df = pd.DataFrame([
        {"Set": name, "Completed Step": completed_step_dict.get(name, None) if name in completed_step_dict else "-"}
        for name in set_names
    ])
    st.subheader("Completed Sets and Completed Steps")
    st.dataframe(set_finish_df)

    st.subheader("How many of each type of card did you get?")
    summary_table = {}
    for ct in card_types:
        summary_table[ct] = {
            "Unique": len(unique_card_type_sets[ct]),
            "Total": len([c for c in all_history if c.startswith(ct)])
        }
    summary_table["Total Card Number"] = {"Unique": len(set(all_history)), "Total": len(all_history)}
    st.write(pd.DataFrame(summary_table).T)



if st.sidebar.button("500 Simulation!"):
    all_total, all_unique = [], []
    sets_complete_counts = []
    set_by_set_counts = {name: [] for name in set_names}
    set_incomplete_missing = {name: [] for name in set_names}
    for i in range(500):
        all_history, unique_card_type_sets, sets_completed_result = simulate_once(
            pack_sequence, unique_first_n=unique_first_n
        )
        this_unique = [len(unique_card_type_sets[ct]) for ct in card_types]
        this_total = [len([c for c in all_history if c.startswith(ct)]) for ct in card_types]
        all_total.append(this_total)
        all_unique.append(this_unique)
        sets_complete_counts.append(sum(sets_completed_result))
        for idx, ok in enumerate(sets_completed_result):
            set_by_set_counts[set_names[idx]].append(1 if ok else 0)
            if not ok:
                eksik = eksik_kart_hesapla_for_set(unique_card_type_sets, set_definitions, card_types, idx)
                set_incomplete_missing[set_names[idx]].append(eksik)


    # --- Kart Türü Sonuçları (Total & Unique, yatay tablo, virgüllü AVG) ---
    stat_cols = ["Avg", "Per 10", "Per 25", "Per 50", "Per 75", "Per 90"]
    # TOTAL
    df_total = pd.DataFrame(all_total, columns=card_types)
    stats_total = pd.DataFrame(index=card_types + ["Total"], columns=stat_cols)
    for i, ct in enumerate(card_types):
        vals = df_total[ct]
        stats_total.loc[ct, "Avg"] = f"{np.mean(vals):.1f}".replace(".", ",")
        stats_total.loc[ct, "Per 10"] = int(np.percentile(vals, 10))
        stats_total.loc[ct, "Per 25"] = int(np.percentile(vals, 25))
        stats_total.loc[ct, "Per 50"] = int(np.percentile(vals, 50))
        stats_total.loc[ct, "Per 75"] = int(np.percentile(vals, 75))
        stats_total.loc[ct, "Per 90"] = int(np.percentile(vals, 90))
        
    # Sütun toplamları
    for col in stat_cols:
        stats_total.loc["Total", col] = sum([float(stats_total.loc[ct, col].replace(",", ".")) if col=="Avg" else int(stats_total.loc[ct, col]) for ct in card_types])
        if col=="Avg":
            stats_total.loc["Total", col] = f"{float(stats_total.loc['Total', col]):.1f}".replace(".", ",")
    st.subheader("Card Selection Results (Total)")
    st.dataframe(stats_total)

    # UNIQUE
    df_unique = pd.DataFrame(all_unique, columns=card_types)
    stats_unique = pd.DataFrame(index=card_types + ["Total"], columns=stat_cols)
    for i, ct in enumerate(card_types):
        vals = df_unique[ct]
        stats_unique.loc[ct, "Avg"] = f"{np.mean(vals):.1f}".replace(".", ",")
        stats_unique.loc[ct, "Per 10"] = int(np.percentile(vals, 10))
        stats_unique.loc[ct, "Per 25"] = int(np.percentile(vals, 25))
        stats_unique.loc[ct, "Per 50"] = int(np.percentile(vals, 50))
        stats_unique.loc[ct, "Per 75"] = int(np.percentile(vals, 75))
        stats_unique.loc[ct, "Per 90"] = int(np.percentile(vals, 90))
        
    for col in stat_cols:
        stats_unique.loc["Total", col] = sum([float(stats_unique.loc[ct, col].replace(",", ".")) if col=="Avg" else int(stats_unique.loc[ct, col]) for ct in card_types])
        if col=="Avg":
            stats_unique.loc["Total", col] = f"{float(stats_unique.loc['Total', col]):.1f}".replace(".", ",")
    st.subheader("Card Selection Results (Unique)")
    st.dataframe(stats_unique)



    sets_df = pd.DataFrame({"Completed Set Count": sets_complete_counts})
    set_quant = sets_df.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T
    set_avg = pd.DataFrame(sets_df.mean()).T
    set_avg.index = ["Avg"]
    st.subheader("Completed Set Stats")
    st.write(pd.concat([set_avg.T, set_quant], axis=1))

  
    # --- Her Setin Kaç Simülasyonda Tamamlandığı + eksik bilgi ---
    set_final = {name: [int(x) for x in set_by_set_counts[name]] for name in set_names}
    set_final_df = pd.DataFrame(set_final)
    set_final_stats = set_final_df.sum(axis=0).to_frame(name="In How Many Simulations Completed")
    set_final_stats["Percent (%)"] = 100 * set_final_stats["In How Many Simulations Completed"] / 500
    set_final_stats["Avg Unique Card on Uncompleted Cards"] = [
        round(np.mean(set_incomplete_missing[name]), 2) if len(set_incomplete_missing[name]) > 0 else 0
        for name in set_names
    ]
    st.subheader("Set Complete Situations")
    st.dataframe(set_final_stats)


