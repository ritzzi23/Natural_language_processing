# Q6: Results

## Table 4: Development and Test Results

| System | Query EM | Record EM | F1 Score |
|--------|----------|-----------|----------|
| **Dev Results** | | | |
| T5 fine-tuned | | | |
| Full model | 3.00% | 82.19% | 83.58% |
| **Test Results** | | | |
| T5 fine-tuning | 0.00% | 68.06% | 70.00% |

**Note**: The model achieved 83.58% Record F1 and 82.19% Record EM on the development set, and 70.00% Record F1 and 68.06% Record EM on the test set. SQL Query Exact Match is low (3.00% on dev, 0.00% on test) because many semantically equivalent SQL queries have different formatting (e.g., spacing, table alias ordering). The test set performance is lower than the development set, which is expected as the test set may contain more challenging examples or different distributions.

---

## Table 5: Qualitative Error Analysis on Dev Set

| Error Type | Example of Error | Error Description | Statistics |
|------------|------------------|-------------------|------------|
| **Missing or Incorrect Condition** | NL: "please give me the flight times i would like to fly from boston to baltimore in the morning before 8"<br><br>GT: `SELECT DISTINCT flight_1.departure_time FROM flight flight_1, ... WHERE ... AND flight_1.departure_time < 800`<br><br>Pred: `SELECT DISTINCT flight_1.arrival_time, flight_1.departure_time FROM flight flight_1, ... WHERE flight_1.departure_time BETWEEN 0 AND 1200 AND ... AND flight_1.departure_time  800` | The model generates incorrect time conditions. In the example, the ground truth uses `departure_time < 800` to represent "before 8 AM", but the model generates `departure_time BETWEEN 0 AND 1200` (incorrect time range) and also has a syntax error (`flight_1.departure_time  800` missing operator). The model struggles with temporal constraints, often generating incorrect time comparisons or missing specific time conditions mentioned in natural language queries. | 69/84 (82.1% of incorrect predictions) |
| **Wrong Column Selection** | NL: "please give me the flight times i would like to fly from boston to baltimore in the morning before 8"<br><br>GT: `SELECT DISTINCT flight_1.departure_time FROM flight ...`<br><br>Pred: `SELECT DISTINCT flight_1.arrival_time, flight_1.departure_time FROM flight ...`<br><br>NL: "what does iah mean"<br><br>GT: `SELECT DISTINCT airport_1.airport_code FROM airport airport_1 WHERE airport_1.airport_code = 'IAH'`<br><br>Pred: `SELECT DISTINCT airline_1.airline_code FROM airline airline_1 WHERE airline_1.airline_code = 'IAH'` | The model selects incorrect columns or tables. In the first example, it adds `arrival_time` when only `departure_time` is requested. In the second example, it queries the `airline` table instead of the `airport` table, selecting `airline_code` instead of `airport_code`. This error occurs when the model misinterprets which entity or attribute is being queried, often confusing similar concepts (e.g., airports vs. airlines, arrival vs. departure times). | 15/84 (17.9% of incorrect predictions) |
| **Wrong Aggregation or Subquery Structure** | NL: "list all flights from boston to san francisco with the maximum number of stops"<br><br>GT: `SELECT DISTINCT flight_1.flight_id FROM flight flight_1, ... WHERE ... AND flight_1.stops = (SELECT MAX(stops) FROM flight flight_1, ... WHERE ...)`<br><br>Pred: `SELECT DISTINCT flight_1.flight_id FROM flight flight_1, ... WHERE ... AND flight_1.stops = (SELECT MAX(flight_1.stops >= 1000))` | The model generates malformed subqueries or incorrect aggregation logic. In the example, the ground truth correctly uses a subquery with `MAX(stops)` to find flights with maximum stops, but the model generates an invalid condition `MAX(flight_1.stops >= 1000)` which is syntactically incorrect. The model struggles with complex aggregation queries, especially those requiring subqueries to find maximum or minimum values, often generating invalid SQL syntax. | 1/84 (1.2% of incorrect predictions) |

**Total incorrect predictions**: 84/466 (18.0%)  
**Total correct predictions**: 382/466 (82.0%)

---

## Analysis Summary

1. **Missing or Incorrect Conditions (82.1% of errors)**: The most common error type involves incorrect WHERE clause conditions, particularly for temporal constraints. The model often generates wrong time comparisons or missing conditions, suggesting difficulty in translating natural language temporal expressions to SQL conditions.

2. **Wrong Column Selection (17.9% of errors)**: The model sometimes selects incorrect columns or queries the wrong table, particularly when similar concepts exist (e.g., airports vs. airlines). This indicates challenges in entity disambiguation and attribute selection.

3. **Wrong Aggregation (1.2% of errors)**: Less common but critical, the model generates malformed subqueries for aggregation operations, particularly for finding maximum/minimum values. This suggests limitations in handling complex nested queries.

**Overall Performance**: Despite these errors, the model achieves 83.58% Record F1, indicating that most queries are semantically correct even if not exactly matching the ground truth SQL format. The low SQL Query EM (3.00%) is expected, as many semantically equivalent queries have different formatting.

