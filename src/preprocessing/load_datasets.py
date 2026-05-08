import os
import pandas as pd
import random
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/intent")
os.makedirs(DATA_DIR, exist_ok=True)


OUR_INTENTS = [
    "inform",       # sharing information
    "question",     # requesting information
    "directive",    # giving instructions
    "commissive",   # making commitments
    "greeting",     # hello, hi, hey
    "farewell",     # bye, goodbye, see you
    "gratitude",    # thank you, thanks
    "complaint",    # this is not working, i am frustrated
    "confirmation", # yes, correct, exactly right
]


DAILY_DIALOG_MAP = {
    0: None,          # dummy — filtered
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive",
}


CLINC_MAP = {
    # question
    "what_is_your_name":       "question",
    "who_made_you":            "question",
    "are_you_a_bot":           "question",
    "how_old_are_you":         "question",
    "what_can_i_ask_you":      "question",
    "time":                    "question",
    "date":                    "question",
    "weather":                 "question",
    "definition":              "question",
    "distance":                "question",
    "currency":                "question",
    "exchange_rate":           "question",
    "calories":                "question",
    "nutrition_info":          "question",
    "recipe":                  "question",
    "restaurant_reviews":      "question",
    "directions":              "question",
    "traffic":                 "question",
    "flight_status":           "question",
    "travel_suggestion":       "question",
    "translate":               "question",
    "fun_fact":                "question",
    "trivia":                  "question",
    "sports":                  "question",
    "current_location":        "question",
    "next_holiday":            "question",
    "confirm_reservation":     "question",
    "vaccines":                "question",
    "user_name":               "question",
    "word_suggestion":         "question",
    "meaning_of_life":         "question",
    "how_busy":                "question",
    "insurance":               "question",
    "pto_balance":             "question",

    # directive
    "reminder":                "directive",
    "reminder_update":         "directive",
    "set_alarm":               "directive",
    "alarm":                   "directive",
    "timer":                   "directive",
    "play_music":              "directive",
    "volume":                  "directive",
    "text":                    "directive",
    "send_email":              "directive",
    "make_call":               "directive",
    "navigate":                "directive",
    "todo_list":               "directive",
    "shopping_list":           "directive",
    "calendar":                "directive",
    "cancel":                  "directive",
    "book_flight":             "directive",
    "book_hotel":              "directive",
    "restaurant_reservation":  "directive",
    "order":                   "directive",
    "uber":                    "directive",
    "schedule_meeting":        "directive",
    "smart_home":              "directive",
    "freeze_account":          "directive",
    "cancel_reservation":      "directive",
    "update_contact":          "directive",
    "change_language":         "directive",
    "reset_settings":          "directive",
    "reorder":                 "directive",
    "todo_list_update":        "directive",
    "shopping_list_update":    "directive",
    "calendar_update":         "directive",
    "update_playlist":         "directive",
    "share_location":          "directive",
    "find_phone":              "directive",
    "change_speed":            "directive",
    "change_volume":           "directive",
    "sync_device":             "directive",

    # commissive
    "maybe":                   "commissive",

    # greeting
    "greeting":                "greeting",

    # farewell
    "goodbye":                 "farewell",

    # gratitude
    "thank_you":               "gratitude",

    # confirmation
    "yes":                     "confirmation",
    "accept":                  "confirmation",
    "confirm":                 "confirmation",

    # complaint
    "account_blocked":         "complaint",
    "report_fraud":            "complaint",
    "report_lost_card":        "complaint",
    "damaged_card":            "complaint",

    # inform
    "tell_joke":               "inform",
    "carry_on":                "inform",
    "oos":                     "inform",
    "repeat":                  "inform",
    "age":                     "inform",
    "location":                "inform",
    "oil_change_how":          "inform",
    "tire_pressure":           "inform",
    "routing":                 "inform",
    "pay_bill":                "inform",
    "bill_balance":            "inform",
    "interest_rate":           "inform",
    "credit_score":            "inform",
    "transactions":            "inform",
    "transfer":                "inform",
    "balance":                 "inform",
    "roll_dice":               "inform",
    "flip_coin":               "inform",
    "taxes":                   "inform",
    "income":                  "inform",
    "spending_history":        "inform",
    "pin_change":              "inform",
    "no":                      "inform",
    "decline":                 "inform",
    "negate":                  "inform",
    "mpg":                     "inform",
    "oil_change_when":         "inform",
    "tire_change":             "inform",
    "jump_start":              "inform",
    "schedule_maintenance":    "inform",
    "replacement_card_duration": "inform",
    "expiration_date":         "inform",
    "min_payment":             "inform",
    "bill_due":                "inform",
    "apr":                     "inform",
    "credit_limit":            "inform",
    "credit_limit_change":     "inform",
    "rewards_balance":         "inform",
    "w2":                      "inform",
    "payday":                  "inform",
    "direct_deposit":          "inform",
    "pto_request":             "inform",
    "pto_request_status":      "inform",
    "pto_used":                "inform",
    "next_song":               "inform",
    "last_song":               "inform",
    "flip_do_not_disturb":     "inform",
    "whisper_mode":            "inform",
    "application_status":      "inform",
    "food_last":               "inform",
    "meal_suggestion":         "inform",
    "gas_type":                "inform",
    "calories":                "inform",
}


SNIPS_MAP = {
    "GetWeather":            "question",
    "SearchCreativeWork":    "question",
    "SearchScreeningEvent":  "question",
    "GetTime":               "question",
    "BookRestaurant":        "directive",
    "RateBook":              "inform",
    "PlayMusic":             "directive",
}


BANKING_MAP = {
        # question
        "balance_not_updated_after_bank_transfer":  "question",
        "balance_not_updated_after_cheque_or_cash_deposit": "question",
        "beneficiary_not_allowed":                  "question",
        "card_about_to_expire":                     "question",
        "card_acceptance":                          "question",
        "card_arrival":                             "question",
        "card_linking":                             "question",
        "card_not_working":                         "question",
        "card_payment_fee_charged":                 "question",
        "card_payment_not_recognised":              "question",
        "card_payment_wrong_exchange_rate":         "question",
        "card_swallowed":                           "question",
        "cash_withdrawal_charge":                   "question",
        "cash_withdrawal_not_recognised":           "question",
        "change_pin":                               "question",
        "compromised_card":                         "question",
        "contactless_not_working":                  "question",
        "country_support":                          "question",
        "declined_card_payment":                    "question",
        "declined_cash_withdrawal":                 "question",
        "declined_transfer":                        "question",
        "direct_debit_payment_not_recognised":      "question",
        "disposable_card_limits":                   "question",
        "exchange_charge":                          "question",
        "exchange_rate":                            "question",
        "exchange_via_app":                         "question",
        "extra_charge_on_statement":                "question",
        "failed_transfer":                          "question",
        "fiat_currency_support":                    "question",
        "get_disposable_virtual_card":              "question",
        "get_physical_card":                        "question",
        "getting_spare_card":                       "question",
        "getting_virtual_card":                     "question",
        "lost_or_stolen_card":                      "question",
        "lost_or_stolen_phone":                     "question",
        "order_physical_card":                      "question",
        "passcode_forgotten":                       "question",
        "pending_card_payment":                     "question",
        "pending_cash_withdrawal":                  "question",
        "pending_top_up":                           "question",
        "pending_transfer":                         "question",
        "pin_blocked":                              "question",
        "receiving_money":                          "question",
        "request_refund":                           "question",
        "reverted_card_payment":                    "question",
        "supported_cards_and_currencies":           "question",
        "terminate_account":                        "question",
        "top_up_by_bank_transfer_charge":           "question",
        "top_up_by_card_charge":                    "question",
        "top_up_by_cash_or_cheque":                 "question",
        "top_up_failed":                            "question",
        "top_up_limits":                            "question",
        "top_up_reverted":                          "question",
        "topping_up_by_card":                       "question",
        "transaction_charged_twice":                "question",
        "transfer_fee_charged":                     "question",
        "transfer_into_account":                    "question",
        "transfer_not_received_by_recipient":       "question",
        "transfer_timing":                          "question",
        "unable_to_verify_identity":                "question",
        "verify_my_identity":                       "question",
        "verify_source_of_funds":                   "question",
        "verify_top_up":                            "question",
        "virtual_card_not_working":                 "question",
        "visa_or_mastercard":                       "question",
        "why_verify_identity":                      "question",
        "wrong_amount_of_cash_received":            "question",
        "wrong_exchange_rate_for_cash_withdrawal":  "question",

        # directive
        "activate_my_card":                         "directive",
        "age_limit":                                "directive",
        "apple_pay_or_google_pay":                  "directive",
        "atm_support":                              "directive",
        "automatic_top_up":                         "directive",
        "beneficiary_not_allowed":                  "directive",

        # complaint
        "card_not_working":                         "complaint",
        "declined_card_payment":                    "complaint",
        "declined_cash_withdrawal":                 "complaint",
        "declined_transfer":                        "complaint",
        "failed_transfer":                          "complaint",
        "transaction_charged_twice":                "complaint",
        "extra_charge_on_statement":                "complaint",
        "wrong_amount_of_cash_received":            "complaint",

        # inform
        "supported_cards_and_currencies":           "inform",
        "visa_or_mastercard":                       "inform",
        "fiat_currency_support":                    "inform",
        "country_support":                          "inform",
        "disposable_card_limits":                   "inform",
        "top_up_limits":                            "inform",
        "cash_withdrawal_charge":                   "inform",
        "top_up_by_bank_transfer_charge":           "inform",
        "top_up_by_card_charge":                    "inform",
        "transfer_fee_charged":                     "inform",
        "exchange_charge":                          "inform",
        "card_payment_fee_charged":                 "inform",
    }


def load_daily_dialog():
    print("Loading DailyDialog...")
    try:
        ds   = load_dataset("benjaminbeilharz/better_daily_dialog")
        data = []
        for split in ds.keys():
            for item in ds[split]:
                text_clean = str(item["utterance"]).strip().lower()
                intent     = DAILY_DIALOG_MAP.get(int(item["turn_type"]))

                if not intent or not text_clean:
                    continue

                if any(text_clean.startswith(g)
                       for g in ["hi ", "hello", "hey ", "hi!", "hello!", "hey!", "hi,"]):
                    intent = "greeting"
                elif any(text_clean.startswith(f)
                         for f in ["bye", "goodbye", "see you", "take care"]):
                    intent = "farewell"
                elif any(w in text_clean
                         for w in ["thank you", "thanks", "thank u"]):
                    intent = "gratitude"
                elif any(text_clean.startswith(c)
                         for c in ["yes,", "yes.", "yes!", "yeah", "correct,", "exactly"]):
                    intent = "confirmation"

                data.append((text_clean, intent))

        print(f"  ✓ {len(data):,} examples loaded")
        return data
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return []       


def load_clinc():
    print("Loading CLINC150...")
    try:
        ds      = load_dataset("clinc_oos", "plus")
        int2lbl = dict(enumerate(ds["train"].features["intent"].names))
        data    = []
        for split in ["train", "validation", "test"]:
            for item in ds[split]:
                intent = CLINC_MAP.get(int2lbl[item["intent"]])
                if intent and item["text"].strip():
                    data.append((item["text"].strip().lower(), intent))
        print(f"  ✓ {len(data):,} examples loaded")
        return data
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return []


def load_snips():
    print("Loading SNIPS...")
    try:
        ds   = load_dataset("snips_built_in_intents")
        data = []
        for split in ds.keys():
            int2lbl = dict(enumerate(ds[split].features["label"].names))
            for item in ds[split]:
                intent = SNIPS_MAP.get(int2lbl[item["label"]])
                if intent and item["text"].strip():
                    data.append((item["text"].strip().lower(), intent))
        print(f"  ✓ {len(data):,} examples loaded")
        return data
    except Exception as e:
        print(f"  ✗ SNIPS failed ({e}), trying alternative...")
        try:
            ds      = load_dataset("benayas/snips")
            int2lbl = dict(enumerate(ds["train"].features["intent"].names))
            data    = []
            for item in ds["train"]:
                intent = SNIPS_MAP.get(int2lbl[item["intent"]])
                if intent and item["text"].strip():
                    data.append((item["text"].strip().lower(), intent))
            print(f"  ✓ {len(data):,} examples loaded (alternative)")
            return data
        except Exception as e2:
            print(f"  ✗ Both failed: {e2}")
            return []


def load_banking77():
    print("Loading Banking77...")
    alternatives = [
        ("legacy-datasets/banking77",       "label"),
        ("PhilipMay/banking77",             "label"),
        ("doyouevenstack/banking77",        "label"),
        ("FinanceInc/auditor_sentiment",    None),   # fallback
    ]

    for repo, label_field in alternatives:
        try:
            ds      = load_dataset(repo)
            split   = "train" if "train" in ds else list(ds.keys())[0]
            features = ds[split].features

            if label_field and label_field in features:
                lf = label_field
            else:
                lf = next((f for f in features if "label" in f.lower() or
                           "intent" in f.lower()), None)
            if not lf:
                continue

            int2lbl = dict(enumerate(ds[split].features[lf].names)) \
                      if hasattr(ds[split].features[lf], "names") else {}
            if not int2lbl:
                continue

            data = []
            for s in ds.keys():
                for item in ds[s]:
                    label  = int2lbl.get(item[lf], "")
                    intent = BANKING_MAP.get(label)
                    text   = item.get("text") or item.get("sentence") or ""
                    if intent and text.strip():
                        data.append((text.strip().lower(), intent))

            if data:
                print(f"  Using: {repo}")
                print(f"  ✓ {len(data):,} examples loaded")
                return data
        except Exception as e:
            print(f"  ✗ {repo} failed: {e}")
            continue

    print("  ✗ All Banking77 alternatives failed")
    return []


def combine_and_save(datasets: dict):
    print("\nCombining datasets...")

    combined = []
    for data in datasets.values():
        combined.extend(data)

    seen, deduped = set(), []
    for text, intent in combined:
        if text not in seen:
            seen.add(text)
            deduped.append((text, intent))

    print(f"  Before dedup : {len(combined):,}")
    print(f"  After dedup  : {len(deduped):,}")
    print(f"  Removed      : {len(combined) - len(deduped):,}")

    random.shuffle(deduped)

    n         = len(deduped)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    splits = {
        "train": deduped[:train_end],
        "val":   deduped[train_end:val_end],
        "test":  deduped[val_end:]
    }

    for name, data in splits.items():
        path = os.path.join(DATA_DIR, f"combined_{name}.csv")
        pd.DataFrame(data, columns=["text", "intent"]).to_csv(path, index=False)
        print(f"  ✓ Saved combined_{name}.csv  ({len(data):,} rows)")

    for name, data in datasets.items():
        path = os.path.join(DATA_DIR, f"{name.lower()}_intent.csv")
        pd.DataFrame(data, columns=["text", "intent"]).to_csv(path, index=False)
        print(f"  ✓ Saved {name.lower()}_intent.csv  ({len(data):,} rows)")

    return splits, deduped


if __name__ == "__main__":
    print("=" * 55)
    print("  LOADING INTENT DATASETS  (9 intents)")
    print("=" * 55)

    daily   = load_daily_dialog()
    clinc   = load_clinc()
    #banking = load_banking77()

    datasets = {}
    if daily:   datasets["DailyDialog"] = daily
    if clinc:   datasets["CLINC150"]    = clinc
    #if banking: datasets["Banking77"]   = banking

    if not datasets:
        print("\nNo datasets loaded. Exiting.")
        exit()

    print(f"\nSuccessfully loaded: {list(datasets.keys())}")
    for name, data in datasets.items():
        from collections import Counter
        dist = Counter(d[1] for d in data)
        print(f"\n  {name}:")
        for intent in OUR_INTENTS:
            print(f"    {intent:<14}: {dist.get(intent, 0):>6,}")

    combine_and_save(datasets)
    print("\nDone. Run analyze_datasets.py next.")