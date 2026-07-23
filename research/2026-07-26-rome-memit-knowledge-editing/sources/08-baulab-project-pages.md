---
url: https://rome.baulab.info/ and https://memit.baulab.info/
title: "ROME and MEMIT project pages (Bau Lab)"
type: project_page
authors: Kevin Meng, David Bau, et al.
year: 2022-2023
accessed: 2026-07-26
quality: 5
relevance: core (primary source, plain-language explanation from authors)
---

Full text extracted via curl (see terminal outputs). Key points already integrated into sources 01 and 02 above. Additional details:

- Causal Tracing mechanism (concrete steps): (1) run the network normally on a factual prompt, recording all hidden states; (2) run again but corrupt the subject token's embeddings with noise ("corrupted run") so the model can no longer recall the fact; (3) selectively restore ("patch back") individual clean hidden states one at a time into the corrupted run, and measure how much doing so restores the correct prediction probability. States whose restoration causes a big jump in correct-answer probability are the "decisive" ones — this reveals the causal (not just correlational) contribution of each hidden state.

- Result of tracing: two causal "sites" — an early site in mid-layer MLPs at the last subject token (this is where facts appear to be "looked up"), and a late site in attention near the final token position (where the information is "delivered" to the output).

- Demonstration example used throughout both project pages: teaching GPT-J the false/counterfactual statement "The Eiffel Tower is located in the city of Rome." After a ROME edit at the early MLP site, the model doesn't just parrot this sentence — it also says, e.g., that traveling to see the Eiffel Tower requires going to Rome, and lists nearby Roman landmarks when asked "what's nearby" — demonstrating true generalization of the injected "belief," not mere text memorization.

- MEMIT extends the single-layer edit to a *range* of layers (e.g., GPT-J layers 3-8) and *spreads* the necessary weight perturbation across all of them, keeping the perturbation per-layer small — this is why MEMIT scales to thousands of edits without destroying general model fluency/performance, whereas naively repeating ROME's single-layer edit for many facts causes catastrophic degradation.
