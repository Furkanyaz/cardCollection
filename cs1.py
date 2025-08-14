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

# --- Star value mapping for the new feature ---
star_values = {
    "1 Star Card": 1,
    "2 Star Card": 2,
    "3 Star Card": 3,
    "4 Star Card": 4,
    "5 Star Card": 5,
    "Gold Card": 5
}

# --- Sidebar: Unique Card Counts Configuration ---
with st.sidebar.expander("Unique Card Counts", expanded=False):
    default_card_counts = {
        "1 Star Card": 17,
        "2 Star Card": 16,
        "3 Star Card": 14,
        "4 Star Card": 13,
        "5 Star Card": 12,
        "Gold Card": 9
    }
    card_counts = {}
    for ct in card_types:
        card_counts[ct] = st.number_input(
            f"{ct} Unique Count",
            min_value=0,
            max_value=100,
            value=default_card_counts[ct],
            step=1,
            key=f"count_{ct}"
        )
# --- Build the full list of unique cards based on these counts ---
unique_card_list = []
for ct in card_types:
    unique_card_list += [f"{ct} {i}" for i in range(card_counts[ct])]

# Default set definitions
default_set_defs = [
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
# Define set names based on defaults
set_names = [f"Set {i+1}" for i in range(len(default_set_defs))]

# --- Sidebar: Dynamic Set Definitions ---
st.sidebar.header("Set Requirements (Unique Cards per Set)")
with st.sidebar.expander("All Set Requirements", expanded=False):
    dynamic_set_defs = []
    for idx, set_name in enumerate(set_names):
        with st.sidebar.expander(set_name, expanded=False):
            req = {}
            for ct in card_types:
                default_val = default_set_defs[idx][ct]
                max_val = card_counts[ct]
                req[ct] = st.number_input(
                    f"{ct}",
                    min_value=0,
                    max_value=max_val,
                    value=default_val,
                    step=1,
                    key=f"set_{idx}_{ct}"
                )
            dynamic_set_defs.append(req)
# Use dynamic definitions below
set_definitions = dynamic_set_defs

# --- Validation: Total unique cards vs. total required by sets ---
total_unique = sum(card_counts.values())
total_required = sum(sum(sdef[ct] for ct in card_types) for sdef in dynamic_set_defs)
if total_unique != total_required:
    st.sidebar.error(
        f"Total unique cards = {total_unique}, but total required by sets = {total_required}. They should match!"
    )
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
    value="M Pack:12, L Pack:10, XL Pack:16, Special Pack:4, Legendary Pack:3"
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


# --- Sidebar: Special/Lengendary Pack 6th‐card probs ---
st.sidebar.header("Special Pack 6th‐Card Probabilities")
special_probs_types = ["3 Star Card", "4 Star Card", "5 Star Card", "Gold Card"]
special_pack_probs = []
for i, ct in enumerate(special_probs_types):
    p = st.sidebar.number_input(
        f"Special Pack → {ct}",
        min_value=0.0, max_value=1.0,
        value=[0.26,0.34,0.32,0.08][i],
        step=0.01, format="%.2f",
        key=f"special6_{i}"
    )
    special_pack_probs.append(p)
if abs(sum(special_pack_probs) - 1.0) > 1e-4:
    st.sidebar.error(f"Special Pack 6th‐card probs must sum to 1.00 (now {sum(special_pack_probs):.2f})")

st.sidebar.header("Legendary Pack 6th‐Card Probabilities")
legendary_pack_probs = []
for i, ct in enumerate(special_probs_types):
    p = st.sidebar.number_input(
        f"Legendary Pack → {ct}",
        min_value=0.0, max_value=1.0,
        value=[0.26,0.34,0.32,0.08][i],
        step=0.01, format="%.2f",
        key=f"legendary6_{i}"
    )
    legendary_pack_probs.append(p)
if abs(sum(legendary_pack_probs) - 1.0) > 1e-4:
    st.sidebar.error(f"Legendary Pack 6th‐card probs must sum to 1.00 (now {sum(legendary_pack_probs):.2f})")

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
    unique_card_type_sets, pack_data=pack_data_dynamic, special_pack_memory=None, legendary_pack_memory=None,
    set_definitions=None, card_types=None
):
    drawn_cards = []
    owned_uniques = set.union(*unique_card_type_sets.values())
    all_unique_left = set(unique_card_list) - owned_uniques

    def draw_one_card(probs_list, types_list):
        mutable_probs = list(probs_list)
        mutable_types = list(types_list)

        while True:
            if not mutable_types or sum(mutable_probs) <= 0:
                return None

            chosen_type = np.random.choice(mutable_types, p=mutable_probs)
            
            pool = [c for c in unique_card_list if c.startswith(chosen_type) and c not in drawn_this_pack]
            if pool:
                return random.choice(pool)
            
            type_index = mutable_types.index(chosen_type)
            mutable_probs.pop(type_index)
            mutable_types.pop(type_index)
            prob_sum = sum(mutable_probs)
            if prob_sum > 0:
                mutable_probs = [p / prob_sum for p in mutable_probs]

    # --- Special/Legendary Pack Logic ---
    if pack_type in ["Special Pack", "Legendary Pack"]:
        for _ in range(5):
            card = draw_one_card(pack_data[pack_type]["prob"], card_types)
            if card:
                drawn_cards.append(card)
                drawn_this_pack.add(card)
        
        special_probs = special_pack_probs if pack_type == "Special Pack" else legendary_pack_probs
        card = draw_one_card(special_probs, special_probs_types)
        if card:
            drawn_cards.append(card)
            drawn_this_pack.add(card)

        # --- START OF FIX: Legendary Pack Guaranteed Unique Card ---
        if pack_type == "Legendary Pack" and unique_guarantee:
            is_any_card_new = any(card not in owned_uniques for card in drawn_cards)
            
            if not is_any_card_new and all_unique_left:
                # All cards are duplicates, let's swap the 6th card.
                if len(drawn_cards) == 6: # Ensure pack is full
                    index_to_replace = 5 # Index of the 6th card
                    new_card = random.choice(list(all_unique_left))
                    
                    # Update drawn_this_pack set carefully
                    old_card = drawn_cards[index_to_replace]
                    if old_card in drawn_this_pack:
                        drawn_this_pack.remove(old_card)
                    
                    drawn_cards[index_to_replace] = new_card
                    drawn_this_pack.add(new_card)
        # --- END OF FIX ---
        
        return drawn_cards

    # --- Regular Pack Logic (M, L, XL) ---
    for _ in range(min(pack_data[pack_type]["count"], unique_give_count)):
        remain_unique = [uc for uc in unique_card_list if uc not in drawn_this_pack and uc not in owned_uniques]
        if not remain_unique: break
        
        remain_counts = {ct: len([uc for uc in remain_unique if uc.startswith(ct)]) for ct in card_types}
        probs = [pack_data[pack_type]["prob"][i] if remain_counts.get(ct, 0) > 0 else 0 for i, ct in enumerate(card_types)]
        
        prob_sum = sum(probs)
        if prob_sum == 0: break
        probs = [p / prob_sum for p in probs]
        
        chosen_type = np.random.choice(card_types, p=probs)
        avail = [uc for uc in remain_unique if uc.startswith(chosen_type)]
        if avail:
            selected = random.choice(avail)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)

    if unique_guarantee and len(drawn_cards) < pack_data[pack_type]["count"]:
        possible_new = list(all_unique_left - set(drawn_this_pack))
        if possible_new:
            selected = random.choice(possible_new)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)

    if gold_guarantee and len(drawn_cards) < pack_data[pack_type]["count"]:
        gold_not_drawn = [c for c in all_unique_left if c.startswith("Gold Card") and c not in drawn_this_pack]
        if gold_not_drawn:
            selected = random.choice(gold_not_drawn)
            drawn_cards.append(selected)
            drawn_this_pack.add(selected)
            
    while len(drawn_cards) < pack_data[pack_type]["count"]:
        card = draw_one_card(pack_data[pack_type]["prob"], card_types)
        if card:
            drawn_cards.append(card)
            drawn_this_pack.add(card)
        else:
            break
            
    return drawn_cards

def track_set_completions_slot_based(unique_card_history, set_definitions, card_types, set_names):
    slots_filled = {s_name: {ct: [] for ct in card_types} for s_name in set_names}
    sets_completed_flags = [False] * len(set_names)
    completion_matrix = []

    for card in unique_card_history:
        card_type = next(ct for ct in card_types if card.startswith(ct))

        for i, set_name in enumerate(set_names):
            sdef = set_definitions[i]
            if sdef[card_type] > len(slots_filled[set_name][card_type]):
                slots_filled[set_name][card_type].append(card)
                break

        for i, set_name in enumerate(set_names):
            sdef = set_definitions[i]
            is_complete = all(len(slots_filled[set_name][ct]) == sdef[ct] for ct in card_types)
            sets_completed_flags[i] = is_complete
        
        completion_matrix.append(list(sets_completed_flags))

    cards_for_viz = {s_name: [] for s_name in set_names}
    for set_name, card_type_dict in slots_filled.items():
        for card_list in card_type_dict.values():
            cards_for_viz[set_name].extend(card_list)
            
    final_matrix_checkmarks = [["✓" if flag else "" for flag in row] for row in completion_matrix]

    return final_matrix_checkmarks, cards_for_viz, slots_filled

def simulate_once(
    pack_sequence, unique_first_n=20, pack_data=pack_data_dynamic
):
    all_history = []
    unique_card_type_sets = {ct: set() for ct in card_types}
    
    special_pack_memory = []
    legendary_pack_memory = []
    
    current_unique_give_count = unique_first_n

    for pack_type in pack_sequence:
        drawn_this_pack = set()
        unique_guarantee = pack_data.get(pack_type, {}).get("new_guarantee", False)
        gold_guarantee = pack_data.get(pack_type, {}).get("gold_guarantee", False)
        is_special = (pack_type == "Special Pack")
        is_legendary = (pack_type == "Legendary Pack")

        drawn_cards = draw_from_pack(
            pack_type, drawn_this_pack,
            unique_give_count=current_unique_give_count,
            unique_guarantee=unique_guarantee,
            gold_guarantee=gold_guarantee,
            unique_card_type_sets=unique_card_type_sets,
            pack_data=pack_data,
            special_pack_memory=special_pack_memory if is_special else None,
            legendary_pack_memory=legendary_pack_memory if is_legendary else None,
            set_definitions=set_definitions, card_types=card_types
        )

        newly_drawn_uniques = 0
        for card in drawn_cards:
            ct = next(ct for ct in card_types if card.startswith(ct))
            if card not in unique_card_type_sets[ct]:
                newly_drawn_uniques +=1

        current_unique_give_count = max(0, current_unique_give_count - newly_drawn_uniques)

        for card in drawn_cards:
            ct = next(ct for ct in card_types if card.startswith(ct))
            all_history.append(card)
            unique_card_type_sets[ct].add(card)
    
    return all_history, unique_card_type_sets
      
pack_sequence = parse_custom_pack_sequence(custom_pack_sequence)

if st.sidebar.button("Start Simulation"):

    all_history, unique_card_type_sets = simulate_once(
        pack_sequence, unique_first_n=unique_first_n
    )

    pack_for_each_card = []
    selection_pack_order_idx = []
    card_counter = 0
    pack_order_counter = 1
    for pack in pack_sequence:
        count = pack_data_dynamic[pack]["count"]
        for _ in range(count):
            if card_counter < len(all_history):
                pack_for_each_card.append(pack)
                selection_pack_order_idx.append(pack_order_counter)
                card_counter += 1
        pack_order_counter += 1

    seen_uniques_for_history = set()
    unique_only_history = []
    for card in all_history:
        if card not in seen_uniques_for_history:
            seen_uniques_for_history.add(card)
            unique_only_history.append(card)
    
    set_completion_matrix, cards_used_for_set_viz, _ = track_set_completions_slot_based(
        unique_only_history, set_definitions, card_types, set_names
    )

    card_rows = []
    seen_uniques = set()
    unique_history_indices = {card: i for i, card in enumerate(unique_only_history)}
    last_status_flags = ["" for _ in set_names]

    for idx, card in enumerate(all_history):
        is_unique = card not in seen_uniques
        seen_uniques.add(card)
        ct = next(ct for ct in card_types if card.startswith(ct))

        if is_unique:
            unique_idx = unique_history_indices.get(card)
            if unique_idx is not None and unique_idx < len(set_completion_matrix):
                last_status_flags = set_completion_matrix[unique_idx]
        
        sets_status = {set_names[i]: last_status_flags[i] for i in range(len(set_names))}
        
        card_rows.append({
            "Card Selection Order": idx + 1,
            "Pack Selection Order": selection_pack_order_idx[idx],
            "Pack": pack_for_each_card[idx],
            "Card": card,
            "Card Type": ct,
            "Total Unique Card": len(seen_uniques),
            "Total Card": idx + 1,
            **sets_status
        })

    st.subheader("All Selected Cards")
    st.dataframe(pd.DataFrame(card_rows))
    
    cumu_rows = []
    cumulative_unique_now = {ct: set() for ct in card_types}
    for idx, card in enumerate(unique_only_history):
        ct = next(ct for ct in card_types if card.startswith(ct))
        cumulative_unique_now[ct].add(card)
        
        status_flags = set_completion_matrix[idx]
        
        original_index = all_history.index(card) if card in all_history else -1
        cumu_rows.append({
            "Unique Card Order": idx + 1,
            "Original Card Order": original_index + 1 if original_index != -1 else "-",
            "Pack": pack_for_each_card[original_index] if original_index != -1 else "-",
            **{f"{ct_name} Unique": len(cumulative_unique_now[ct_name]) for ct_name in card_types},
            **{set_names[i]: status_flags[i] for i in range(len(set_names))}
        })
    
    st.subheader("Cumulative Card Selection Table (Unique Cards Only)")
    st.dataframe(pd.DataFrame(cumu_rows))

    completed_info = []
    for set_idx, set_name in enumerate(set_names):
        completed_step = None
        completed_pack_order = None
        for i, flags in enumerate(set_completion_matrix):
            if flags[set_idx] == "✓":
                completed_step = i + 1
                card_that_completed = unique_only_history[i]
                original_card_idx = all_history.index(card_that_completed) if card_that_completed in all_history else -1
                if original_card_idx != -1:
                    completed_pack_order = selection_pack_order_idx[original_card_idx]
                break
        completed_info.append({
            "Set": set_name,
            "Completed at Unique Step": completed_step if completed_step is not None else "-",
            "Completed in Pack Order": completed_pack_order if completed_pack_order is not None else "-"
        })
    
    set_finish_df = pd.DataFrame(completed_info)
    st.subheader("Completed Sets and Completed Steps")
    st.dataframe(set_finish_df)

    st.subheader("How many of each type of card did you get?")
    summary_table = {}
    total_stars = 0
    for ct in card_types:
        total_cards_of_type = len([c for c in all_history if c.startswith(ct)])
        stars_for_type = total_cards_of_type * star_values[ct]
        total_stars += stars_for_type
        summary_table[ct] = {
            "Unique": len(unique_card_type_sets[ct]),
            "Total": total_cards_of_type,
            "Stars": stars_for_type
        }
    summary_table["Total Card Number"] = {
        "Unique": len(set(all_history)), 
        "Total": len(all_history),
        "Stars": total_stars
    }
    st.write(pd.DataFrame(summary_table).T)

    st.subheader("Set Slots")
    for idx, set_name in enumerate(set_names):
        st.write(f"**{set_name}**")
        assigned = cards_used_for_set_viz[set_name]
        html = ""
        for ct, req_count in set_definitions[idx].items():
            if req_count == 0: continue
            filled = [c for c in assigned if c.startswith(ct)]
            for i in range(req_count):
                is_filled = i < len(filled)
                bg = "#e6ffe6" if is_filled else "#ffe6e6"
                label = filled[i].split(" ")[-1] if is_filled else ""
                card_name = filled[i] if is_filled else f"({ct} Eksik)"
                star_display = 'GOLD' if ct == 'Gold Card' else '★' * int(ct.split()[0])
                html += (
                    f"<div style='display:inline-block;width:80px;height:110px;"
                    f"background:{bg};border:2px solid #888;border-radius:6px;"
                    f"margin:4px;text-align:center;font-size:12px;padding-top:5px;box-shadow: 2px 2px 5px rgba(0,0,0,0.2);' title='{card_name}'>"
                    f"<div style='color:gold;font-size:14px;font-weight:bold;'>{star_display}</div>"
                    f"<div style='font-weight:bold;font-size:14px;margin-top:10px;'>{label}</div>"
                    f"<div style='font-size:10px;margin-top:10px;color:#555;'>{card_name if is_filled else ''}</div>"
                    f"</div>"
                )
        st.write(html, unsafe_allow_html=True)


if st.sidebar.button("500 Simulation!"):
    all_total_counts = []
    all_unique_counts = []
    all_star_counts = []
    all_sets_completed_counts = []
    per_set_completion_counts = {name: 0 for name in set_names}
    per_set_missing_cards = {name: [] for name in set_names}

    progress_bar = st.progress(0, text="Simulations running...")

    for i in range(500):
        all_history, unique_card_type_sets = simulate_once(
            pack_sequence, unique_first_n=unique_first_n
        )

        run_stars = [len([c for c in all_history if c.startswith(ct)]) * star_values[ct] for ct in card_types]
        all_star_counts.append(run_stars)

        all_total_counts.append([len([c for c in all_history if c.startswith(ct)]) for ct in card_types])
        all_unique_counts.append([len(unique_card_type_sets[ct]) for ct in card_types])

        seen_uniques_for_history = set()
        unique_only_history = []
        for card in all_history:
            if card not in seen_uniques_for_history:
                seen_uniques_for_history.add(card)
                unique_only_history.append(card)
        
        completion_matrix, _, final_slots_filled = track_set_completions_slot_based(
            unique_only_history, set_definitions, card_types, set_names
        )

        if completion_matrix:
            final_status_bool = completion_matrix[-1]
            final_status_int = [1 if flag == "✓" else 0 for flag in final_status_bool]
            all_sets_completed_counts.append(sum(final_status_int))

            for idx, name in enumerate(set_names):
                per_set_completion_counts[name] += final_status_int[idx]
                if final_status_int[idx] == 0:
                    sdef = set_definitions[idx]
                    missing_count = 0
                    for ct in card_types:
                        required = sdef[ct]
                        filled = len(final_slots_filled[name][ct])
                        missing_count += max(0, required - filled)
                    per_set_missing_cards[name].append(missing_count)
        else:
            all_sets_completed_counts.append(0)
            for idx, name in enumerate(set_names):
                sdef = set_definitions[idx]
                missing_count = sum(sdef.values())
                if missing_count > 0:
                    per_set_missing_cards[name].append(missing_count)

        progress_bar.progress((i + 1) / 500, text=f"Simulations running... ({(i + 1)}/500)")

    stat_cols = ["Avg", "Per 10", "Per 25", "Per 50", "Per 75", "Per 90"]
    
    st.subheader("Card Selection Results (Total) - 500 Simulations")
    df_total = pd.DataFrame(all_total_counts, columns=card_types)
    stats_total = pd.DataFrame(index=card_types + ["Total"], columns=stat_cols)
    for ct in card_types:
        vals = df_total[ct]
        stats_total.loc[ct, "Avg"] = f"{np.mean(vals):.1f}".replace(".", ",")
        stats_total.loc[ct, "Per 10"] = int(np.percentile(vals, 10))
        stats_total.loc[ct, "Per 25"] = int(np.percentile(vals, 25))
        stats_total.loc[ct, "Per 50"] = int(np.percentile(vals, 50))
        stats_total.loc[ct, "Per 75"] = int(np.percentile(vals, 75))
        stats_total.loc[ct, "Per 90"] = int(np.percentile(vals, 90))
    for col in stat_cols:
        total_val = sum([float(str(stats_total.loc[ct, col]).replace(",", ".")) for ct in card_types])
        stats_total.loc["Total", col] = f"{total_val:.1f}".replace(".", ",") if col == "Avg" else int(total_val)
    st.dataframe(stats_total)

    st.subheader("Card Selection Results (Unique) - 500 Simulations")
    df_unique = pd.DataFrame(all_unique_counts, columns=card_types)
    stats_unique = pd.DataFrame(index=card_types + ["Total"], columns=stat_cols)
    for ct in card_types:
        vals = df_unique[ct]
        stats_unique.loc[ct, "Avg"] = f"{np.mean(vals):.1f}".replace(".", ",")
        stats_unique.loc[ct, "Per 10"] = int(np.percentile(vals, 10))
        stats_unique.loc[ct, "Per 25"] = int(np.percentile(vals, 25))
        stats_unique.loc[ct, "Per 50"] = int(np.percentile(vals, 50))
        stats_unique.loc[ct, "Per 75"] = int(np.percentile(vals, 75))
        stats_unique.loc[ct, "Per 90"] = int(np.percentile(vals, 90))
    for col in stat_cols:
        total_val = sum([float(str(stats_unique.loc[ct, col]).replace(",", ".")) for ct in card_types])
        stats_unique.loc["Total", col] = f"{total_val:.1f}".replace(".", ",") if col == "Avg" else int(total_val)
    st.dataframe(stats_unique)

    st.subheader("Star Collection Results - 500 Simulations")
    df_stars = pd.DataFrame(all_star_counts, columns=card_types)
    stats_stars = pd.DataFrame(index=card_types + ["Total"], columns=stat_cols)
    for ct in card_types:
        vals = df_stars[ct]
        stats_stars.loc[ct, "Avg"] = f"{np.mean(vals):.1f}".replace(".", ",")
        stats_stars.loc[ct, "Per 10"] = int(np.percentile(vals, 10))
        stats_stars.loc[ct, "Per 25"] = int(np.percentile(vals, 25))
        stats_stars.loc[ct, "Per 50"] = int(np.percentile(vals, 50))
        stats_stars.loc[ct, "Per 75"] = int(np.percentile(vals, 75))
        stats_stars.loc[ct, "Per 90"] = int(np.percentile(vals, 90))
    for col in stat_cols:
        total_val = sum([float(str(stats_stars.loc[ct, col]).replace(",", ".")) for ct in card_types])
        stats_stars.loc["Total", col] = f"{total_val:.1f}".replace(".", ",") if col == "Avg" else int(total_val)
    st.dataframe(stats_stars)

    st.subheader("Completed Set Stats (Total number of completed sets out of 9)")
    sets_df = pd.DataFrame({"Completed Set Count": all_sets_completed_counts})
    set_quantiles = sets_df["Completed Set Count"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    set_stats = pd.DataFrame({
        "Avg": [f"{sets_df['Completed Set Count'].mean():.1f}"],
        "Per 10": [int(set_quantiles[0.1])],
        "Per 25": [int(set_quantiles[0.25])],
        "Per 50": [int(set_quantiles[0.5])],
        "Per 75": [int(set_quantiles[0.75])],
        "Per 90": [int(set_quantiles[0.9])],
    }, index=["Count"])
    st.dataframe(set_stats)
    
    st.subheader("Per-Set Completion Statistics")
    per_set_df = pd.DataFrame.from_dict(per_set_completion_counts, orient='index', columns=['Completion Count'])
    per_set_df['Completion Rate'] = (per_set_df['Completion Count'] / 500 * 100).map('{:.1f}%'.format)
    avg_missing = {}
    for name, missing_list in per_set_missing_cards.items():
        if missing_list:
            avg_missing[name] = f"{np.mean(missing_list):.1f}"
        else:
            avg_missing[name] = "0.0"
    per_set_df["Ort. Eksik Kart (Tamamlanamayanlarda)"] = pd.Series(avg_missing)
    st.dataframe(per_set_df)
