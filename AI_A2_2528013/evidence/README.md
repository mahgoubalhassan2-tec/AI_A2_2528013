# Required demonstration evidence

The assignment requires one continuous, unedited MP4 file named `AI_A2_2528013_DEMO.mp4`, no longer than two minutes. This archive does not contain a fabricated placeholder video. You must record the demonstration on your own computer so it truthfully shows your identity and actual execution.

## Recording checklist
1. Start with the terminal clock visible and show a handwritten card with your full name, registration number `25/28013`, and verification code `AI-A2-8013`.
2. Show the project folder and the real final commit hash if you use a Git repository.
3. Run `python run_all.py --student-id 25/28013 --seed 8013`; show the fingerprint and generated artifact list.
4. Open `artifacts/regression_metrics.json` and one plot. Explain one regression error in your own words.
5. Run `python predict.py --json samples/valid.json`, then `python predict.py --json samples/invalid.json` and show the validation failure.
6. Start `uvicorn app:app --host 127.0.0.1 --port 8000`; open `http://127.0.0.1:8000/docs`; demonstrate `/health`, valid `/predict`, and a validation error.
7. Run `python -m pytest -q`; show the report and declaration, then show the terminal clock again.

Keep the recording continuous and unedited, and follow the official brief. The student must personally explain one classification error and one cluster profile as well as the regression error.
