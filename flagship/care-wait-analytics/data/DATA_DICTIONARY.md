# Data dictionary (all data is synthetic)

| Column | Type | Meaning |
|---|---|---|
| referral_id | text | Unique referral identifier, format REF-00001 |
| referral_date | date | Date the referral was received, Jan to Jun 2026 |
| specialty | text | One of Orthopaedics, Ophthalmology, ENT, Dermatology, Cardiology, General Surgery |
| priority | text | routine or urgent |
| age_band | text | 0-17, 18-39, 40-59, 60-74, 75+ |
| imd_quintile | int | Deprivation quintile 1 (least deprived) to 5 (most deprived) |
| wait_days | int | Days from referral to treatment or current wait, 5 to 400 in clean data |
| treatment_date | date | referral_date + wait_days |
| breach_18wk | int | 1 if wait_days is over 126 days (18 weeks), else 0 |

The raw generator adds 75 deliberately dirty rows (30 duplicates,
20 missing specialty, 15 negative waits, 10 invalid age bands) so the
quality layer has something real to catch. Clean output is 6,000 rows.
