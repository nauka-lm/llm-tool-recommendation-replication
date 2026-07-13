# Independent Second-Rater Instructions

**Study:** Inter-rater reliability check for the manuscript "Hallucination Mitigation in Large Language Model-Based Tool Recommendation: A Cross-Provider Architectural Ablation Study Across Two Model Generations" (Manuscript ID ai-4346677, journal *AI*, MDPI), authors L. Menxhiqi and G. Marinova.

**Purpose.** In the manuscript, 197 tool names produced by language models were classified into four categories by the first author. A peer reviewer has requested that a second person independently repeat this classification so that inter-rater agreement (Cohen's kappa) can be computed and reported. You are that independent second rater. Your classifications will be compared with the first author's, disagreements will be resolved by discussion afterwards, and the agreement statistic will be reported in the manuscript.

**Independence requirement (essential).** You must complete the sheet using your own judgment only. You have deliberately NOT been shown the first author's classifications, and you should not ask about them or try to guess them. Disagreement on some names is normal and expected; it is part of the method. An honest independent judgment is what makes this check scientifically valid.

---

## Background, in one paragraph

Our platform (Online-CADCOM) contains a verified inventory of 82 engineering software tools (printed circuit board design, circuit simulation, power electronics and related areas). In our experiments, language models sometimes recommended tool names that are NOT in that inventory. The 197 names in your sheet are the most frequent of those. Your task is to determine, for each name, what it actually is.

## The materials you received

| File | Purpose |
|---|---|
| `IRR-Rating-Sheet-BLIND.xlsx` | The sheet you fill in: 197 rows, a dropdown in the Category column, and an optional Notes column. (A CSV copy is included in case you prefer it.) |
| `inventory-reference.txt` | The list of tool names our platform recognizes as in-inventory, including known aliases. Used only for the "Near-miss" decision. |
| `RATER-INSTRUCTIONS.pdf` | This document. |

## The four categories

Assign exactly one category to each name:

| Category | Meaning | How you recognize it |
|---|---|---|
| **H2** | A real commercial or open-source tool that exists in the world but is not in our inventory. | A web search for the name finds a genuine product page, vendor site, or code repository. Examples: LTspice, ANSYS, COMSOL. |
| **Near-miss** | A variant, truncation, or partial form of a tool that IS in our inventory. | The name clearly corresponds to an entry (or part of an entry) in `inventory-reference.txt`. Example: "Saturn" -> *Saturn PCB Design Toolkit*. |
| **Non-specific** | A generic label, phrase, workflow word, heading, or technical term that does not refer to any specific product. | No search needed; it is visibly not a product name. Examples: "Practical Tips", "Recommended", "Score", "Layer". |
| **H1** | A fabricated tool name: it sounds like a product but has no real-world referent. | A web search finds no such product, and the name is not a generic phrase. Example: "PCB Pro Designer 3000". |

## Procedure per name (fastest order)

1. Ask: *is this even a product name?* If it is clearly a generic word or phrase, select **Non-specific** and move on (no search needed).
2. Otherwise, search the name on the web (your own browser, any search engine). If you find a genuine product with that name, select **H2** — unless the product you found is actually one of ours (check `inventory-reference.txt`), in which case select **Near-miss**.
3. If the name looks like a fragment or variant of an inventory entry, check `inventory-reference.txt`; if it matches, select **Near-miss**.
4. If you find no real product and the name is not generic, select **H1**.

Guidelines:
- Spend at most about two minutes per name. If you remain unsure, choose your best answer and write "unsure" in the Notes column.
- The whole task takes roughly 1.5 to 3 hours and can be split over several sittings (the file saves normally).
- Please do not use AI assistants to decide categories; the reviewer's request is specifically for independent human judgment. Ordinary web search is expected and appropriate.

## When you are done

1. Confirm every one of the 197 rows has a category.
2. Save the file, adding your initials to the file name (e.g., `IRR-Rating-Sheet-BLIND-XY.xlsx`).
3. Return it by email to the first author (lavdim.menxhiqi@ubt-uni.net), and in the same email please confirm the following declaration in your own words or as written:

> *"I completed the rating sheet independently, using my own judgment and my own web searches. I was not shown, and did not seek, the first author's classifications before or during the rating."*

Your email reply serves as the formal record of the independent rating.

## What happens with your ratings

Your labels will be compared with the first author's; Cohen's kappa with a confidence interval will be computed and reported in the manuscript. Any disagreements will be discussed together and resolved by consensus, and the number of changed labels will be reported. With your permission, we would like to thank you by name in the Acknowledgements section of the article; please state in your reply whether you agree.

Thank you very much for your time and care.
