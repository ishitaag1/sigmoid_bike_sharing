"""
Demand assistant: a simple prototype for the operations manager.

Ask a question in plain English, e.g.
    How many bikes do we need at 8am on Monday?
    Busiest hours on a rainy Saturday?
    Will 300 bikes be enough at 6pm on Friday in July?

How it works:
    1. Look for keywords in the question (hour, day, month, weather, fleet size).
    2. Fill anything missing with a default, and say so in the answer.
    3. Ask the demand model (linear regression from the notebook) for the numbers.

Run:  python demand_assistant.py
"""

import re
import pandas as pd
from model_utils import load_and_clean, train_model, predict, bikes_needed, PLANNING_BUFFER

DAYS = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]  # dataset order: 0 = Sunday
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
SEASON_OF_MONTH = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}

print("Loading data and training the model...")
df = load_and_clean("hour.csv")
model = train_model(df)
typical_weather = df.groupby("mnth")[["temp_c", "hum_pct", "wind_kmh"]].median()


def read_question(q):
    """Pull the details out of the question. Returns the settings and a list of defaults used."""
    q = q.lower()
    defaults = []

    # Hour: "8am", "6 pm", "17:00"
    hour = None
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", q)
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
    m2 = re.search(r"\b(\d{1,2}):00\b", q)
    if hour is None and m2:
        hour = int(m2.group(1))

    # Day of week
    weekday = next((i for i, d in enumerate(DAYS) if d in q), None)
    if weekday is None and "weekend" in q:
        weekday = 6
    if weekday is None:
        weekday = 3
        defaults.append("Wednesday")

    # Month
    month = next((i + 1 for i, mo in enumerate(MONTHS) if mo in q), None)
    if month is None:
        month = 6
        defaults.append("June")

    # Weather
    if any(w in q for w in ["rain", "snow", "storm"]):
        weather = 3
    elif any(w in q for w in ["cloud", "mist", "fog"]):
        weather = 2
    else:
        weather = 1
        defaults.append("clear weather")

    # Fleet size: "300 bikes"
    m3 = re.search(r"(\d+)\s*bikes", q)
    fleet = int(m3.group(1)) if m3 else None

    holiday = int("holiday" in q)
    settings = {"hour": hour, "weekday": weekday, "month": month, "weather": weather,
                "holiday": holiday, "fleet": fleet}
    return settings, defaults


def forecast_day(s):
    """Predict all 24 hours for the given settings (at 2012 demand levels)."""
    w = typical_weather.loc[s["month"]]
    workingday = int(s["weekday"] not in (0, 6) and not s["holiday"])
    rows = pd.DataFrame({
        "hr": range(24),
        "workingday": workingday,
        "weekday": s["weekday"],
        "holiday": s["holiday"],
        "mnth": s["month"],
        "season": SEASON_OF_MONTH[s["month"]],
        "yr": 1,
        "weathersit": s["weather"],
        "temp_c": w.temp_c,
        "hum_pct": 85 if s["weather"] == 3 else w.hum_pct,
        "wind_kmh": w.wind_kmh,
    })
    rows["hr_day"] = rows.hr.astype(str) + "_" + rows.workingday.astype(str)
    rows["expected"] = predict(model, rows)
    rows["plan"] = rows.expected * PLANNING_BUFFER
    return rows, workingday


def answer(question):
    s, defaults = read_question(question)
    day, workingday = forecast_day(s)
    weather_name = {1: "clear", 2: "cloudy", 3: "rain/snow"}[s["weather"]]
    lines = [f"[{DAYS[s['weekday']].title()}, {MONTHS[s['month'] - 1].title()}, {weather_name}]"]

    if s["hour"] is not None:
        r = day.iloc[s["hour"]]
        need = bikes_needed(r.plan)
        lines.append(f"At {s['hour']:02d}:00 expect ~{r.expected:.0f} rentals "
                     f"(plan for up to {r.plan:.0f}). Keep ~{need:.0f} bikes available.")
        if s["fleet"]:
            ok = "should be enough" if s["fleet"] >= need else "is probably NOT enough"
            lines.append(f"{s['fleet']} bikes {ok} system-wide (they still need to be at the right stations).")
    else:
        top = day.nlargest(3, "expected")
        lines.append(f"Expected total for the day: ~{day.expected.sum():,.0f} rentals.")
        lines.append("Busiest hours: " + ", ".join(f"{h:02d}:00 (~{v:.0f})" for h, v in zip(top.hr, top.expected)))
        lines.append(f"At the peak, keep ~{bikes_needed(day.plan.max()):.0f} bikes available.")

    if workingday:
        lines.append("Tip: fill residential stations before 7 AM, clear full downtown docks mid-morning, "
                     "and keep enough bikes downtown for the 5 PM rush.")
    else:
        lines.append("Tip: leisure day. Demand builds from late morning, so stock parks and leisure spots early.")

    if defaults:
        lines.append("Assumed: " + ", ".join(defaults) + " (typical temperature for the month).")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Ask about bike demand (type 'quit' to stop).")
    print("Try: How many bikes do we need at 8am on Monday in September?\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("\n" + answer(q) + "\n")
