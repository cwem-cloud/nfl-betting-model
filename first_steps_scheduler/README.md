# First Steps SLP Schedule Optimizer

Generates an optimised weekly schedule for a Speech-Language Pathologist doing
in-home pediatric therapy through Indiana's First Steps program.  The tool
minimises total drive time while respecting each family's availability and
produces three lunch-break variants so the clinician can compare options.

---

## Setup

### 1. Clone / enter the project directory

```bash
cd first_steps_scheduler
```

### 2. Install dependencies

Python 3.11+ is required.

```bash
pip install -r requirements.txt
```

### 3. Get a free OpenRouteService API key

1. Sign up at <https://openrouteservice.org/dev/#/signup>
2. Copy your token from the dashboard.

### 4. Configure the API key

Copy `.env.example` to `.env` and paste your key:

```bash
cp .env.example .env
# then edit .env
ORS_API_KEY=paste_your_key_here
```

Alternatively, paste the key directly into the sidebar inside the app.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## CSV Format

Upload a file with these columns (order does not matter):

| Column | Required | Description |
|---|---|---|
| `client_id` | ✅ | Anonymised identifier, e.g. "Client A" |
| `intersection` | ✅ | Nearest cross-street or address, e.g. "86th St & Ditch Rd, Indianapolis IN" |
| `availability` | ✅ | Free-text using the DSL described below |
| `notes` | ✗ | Optional free-text notes (not used by the optimizer) |

See `data/sample_clients.csv` for a ready-to-use example with 14 clients.

---

## Availability DSL

Write availability in a plain-English mini-language inside the `availability`
column.  The parser is forgiving with capitalisation and whitespace.

### All days, no restrictions

```
any
```

### Single day

```
Thu only
Mon
Tuesday
```

### Multiple days (comma-separated)

```
Mon, Wed
Tue, Thu
```

### Day range

```
M-W
Mon-Thu
Tue-Thu
```

### Time floor ("after …")

```
Mon after 2pm
any after 9am
M-W after 1:30pm
```

### Time ceiling ("before …")

```
Tue before noon
Mon, Wed before 3pm
Thu before 12:30pm
```

### Combining days and times

```
Tue, Thu before noon
M-W after 2pm
Mon after 9am
```

### Period keywords

```
Mon mornings        → Mon, window ends at noon
Thu afternoons      → Thu, window starts at noon
```

### Exclusion ("not …")

```
not Mon             → all days except Monday
not Mon mornings    → all days; Monday afternoon only
not Thursday
```

### Notes

* Working hours are **8:30 AM – 4:00 PM**, Monday–Thursday only.
* All sessions are **60 minutes**.
* Windows that fall entirely outside working hours produce an error.
* When a client's time window is too narrow to fit any session after travel
  constraints, they appear in the **Unscheduled Clients** warning.

---

## App Workflow

1. **Upload CSV** (or click "Use Sample Data")
2. Review parsed clients and availability windows in the table
3. Click **Geocode Addresses** – pins appear on a preview map for verification
4. Click **Generate All Three Variants** – the optimizer runs three times
5. Compare results across three tabs:
   - **No Lunch** – back-to-back sessions
   - **Daily Lunch** – 30-minute break between 11:30 AM and 1:30 PM every day
   - **Hybrid** – lunch on 2 of the 4 days (optimizer picks the best 2)
6. Download any variant as CSV with the **Download CSV** button

---

## Architecture

```
first_steps_scheduler/
├── app.py                  # Streamlit UI (all UI logic)
├── scheduler/
│   ├── __init__.py
│   ├── parser.py           # Availability DSL → TimeWindow objects
│   ├── geocoder.py         # ORS geocoding wrapper + geocode_cache.json
│   ├── distance_matrix.py  # ORS matrix API + matrix_cache.json
│   ├── optimizer.py        # OR-Tools VRPTW solver (3 variants)
│   └── exporter.py         # DataFrame, Folium map, CSV bytes
├── data/
│   ├── sample_clients.csv
│   ├── geocode_cache.json  # auto-created on first geocode
│   └── matrix_cache.json   # auto-created on first matrix build
├── tests/
│   └── test_parser.py      # 26 pytest cases for the DSL parser
├── .env.example
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Caching

| Cache file | What it stores | Key |
|---|---|---|
| `data/geocode_cache.json` | `address → [lon, lat]` | Normalised address string |
| `data/matrix_cache.json` | `hash → NxN matrix` | SHA-256 of location list |

Both caches are keyed so the same address or roster never hits the API twice.
Delete a cache file to force fresh lookups.

---

## Extending the Optimizer

See the module-level docstring in `scheduler/optimizer.py` for a full
explanation of the OR-Tools model.  Key extension points:

* **More vehicles / days** – change `num_vehicles` and `DAYS` in `optimizer.py`
  and `parser.py`.
* **Different session lengths** – adjust `SESSION_MINUTES`.
* **Harder lunch constraints** – change `is_optional=True` to `False` in
  `_add_lunch_constraints`.
* **Longer solver time** – increase `SOLVER_TIME_LIMIT_SEC`.
