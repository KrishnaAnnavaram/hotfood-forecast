# ASD-STE100 Simplified Technical English: the standard for this repository

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. Rules for the text

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, "test" is a noun or a verb, "check" is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: "prepare", "do", "find", "get", "make".
4. Do not use an "-ing" form as a noun or an adjective ("the running job", "after indexing").
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and "can" for a possibility.
8. Keep the articles "a", "an" and "the" in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: "If the index is stale, build it again."
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase ("The cost model") or an imperative ("Run the demo").
   Do not start a heading with an "-ing" form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or "check that" |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the hotfood-forecast documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **sales file** | The CSV file with one row per day in the store layout (`HOTFOOD_DATA`). | dataset (for the file), table, sheet |
| **category** | One of the four forecast groups: `Chicken`, `FriedSnacks`, `FriedBurritos`, `Otherfood`. | item, product, target group |
| **units** | The number of items sold in one category on one day. | quantity (for sales), volume |
| **demand** | The number of units that customers want on one day. The recorded units are a lower limit of the demand. | sales (for demand), need |
| **closure day** | A day when the store is closed. All four categories have zero units, or `Closed` is 1. | zero day, holiday (for closure) |
| **calendar gap** | A date between the first and the last date that has no usable row in the sales file. | missing value, hole |
| **observed day** | A day that is not a closure day and not a calendar gap. | valid day, open day |
| **origin** | The last day with known units when a forecast is made. | cutoff (in prose), anchor, as-of date |
| **horizon** | The number of days from the origin to the target date (1 to 28). | lead time, step, lag |
| **target date** | The day that a forecast is for: origin + horizon. | forecast date, future day |
| **feature row** | One row of features for one category, one origin and one horizon. | sample, instance, record |
| **demand feature** | A feature that reads units on or before the origin. | lag feature (in prose), history feature |
| **calendar feature** | A feature that describes the target date: weekday, month, holiday flag. | date feature, seasonality feature |
| **holiday calendar** | The rule-based list of holidays (`us_federal` or `none`). | holiday file, holiday flag (for the list) |
| **model** | One forecast method from the model list: `seasonal_naive`, `weekday_mean`, `ridge`, `gbm` or `lightgbm`. | algorithm, learner, estimator (in prose) |
| **baseline** | The model `seasonal_naive` or `weekday_mean`. A baseline has no trained parameters. | benchmark, naive model |
| **pipeline** | A scikit-learn `Pipeline` that holds the preprocessing and the model. | workflow, chain |
| **quantile forecast** | The forecast value at one probability level, for example `q0.9`. | percentile, band |
| **point forecast** | The quantile forecast at 0.5 (`q0.5`). | prediction (alone), estimate |
| **backtest** | The rolling-origin test of models on past origins. | cross-validation, simulation (for the test) |
| **fold** | One origin of the backtest with its training rows and its test rows. | split, window |
| **final model** | A new model fit on all observed days. The forecast and the order use it. | production model, last model |
| **order** | The number of units to cook for one category on one target date. | order quantity (in prose), plan |
| **policy** | The rule that turns quantile forecasts into an order: `newsvendor` or `p50`. | strategy, method |
| **critical ratio** | The value cu / (cu + co) for a category. | service level, target quantile |
| **underage cost** | The money lost for each unit of demand that has no stock: price - unit cost. | shortage cost, lost sale |
| **overage cost** | The money lost for each wasted unit: unit cost + disposal cost. | waste cost (alone), holding cost |
| **waste** | Units that are cooked but not sold on the same day. | spoilage, leftover, shrink |
| **stock-out** | Demand that the order cannot supply. | shortage, lost demand |
| **WAPE** | Weighted absolute percentage error: sum of absolute errors divided by the sum of units. | MAPE, percentage error |
| **pinball loss** | The quantile loss of a quantile forecast, averaged over the quantiles. | quantile score, check loss |
| **coverage** | The fraction of actual units inside the outer quantile interval. | hit rate, calibration (alone) |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **validate** | Check the sales file against the schema and give the validation report. |
| **flag** | Mark a day as a closure day or a calendar gap. The day stays in the calendar. |
| **build** | Make feature rows from the sales file. |
| **fit** | Train a model or a pipeline on training rows. |
| **forecast** | Give quantile forecasts for target dates after an origin. |
| **score** | Compare forecasts with the actual units and calculate the metrics. |
| **simulate** | Calculate units sold, waste, stock-outs and profit for a policy on backtest days. |
| **order** | Calculate the order for a category and a target date. |
