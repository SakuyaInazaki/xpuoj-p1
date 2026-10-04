# P1 platform state — 2026-09-12

- Contest 13 problem 1 uses language ID `triton-dist`.
- Read-only `judgeClient/checkAvailability` with
  `requiredFlags=["custom-test"]` returned `{"available":false}`.
- `getSessionInfo.availableLanguages.customTestModes` has no `triton-dist`
  entry. P1 distributed Triton custom tests are therefore unavailable today.
- Current problem and language metadata expose no Triton or image version
  field. The remote P1 Triton version and any image upgrade remain unknown.
- P1 scoreboard remains 72.75, best SID 141408, submission count 2589.
  Latest SID 141474 is terminal; there are no P1 submissions in flight.
- These checks were read only. No submission or custom-test token was used.
