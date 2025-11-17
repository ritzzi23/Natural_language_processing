# Q6: Results - Table Format for LaTeX

## Table 4: Development and Test Results

```
\begin{tabular}{|l|c|c|c|}
\hline
System & Query EM & Record EM & F1 Score \\
\hline
\textbf{Dev Results} & & & \\
T5 fine-tuned & & & \\
Full model & 3.00\% & 82.19\% & 83.58\% \\
\hline
\textbf{Test Results} & & & \\
T5 fine-tuning & 0.00\% & 68.06\% & 70.00\% \\
\hline
\end{tabular}
```

**Plain text format:**
```
| System | Query EM | Record EM | F1 Score |
|--------|----------|-----------|----------|
| Dev Results | | | |
| T5 fine-tuned | | | |
| Full model | 3.00% | 82.19% | 83.58% |
| Test Results | | | |
| T5 fine-tuning | 0.00% | 68.06% | 70.00% |
```

---

## Table 5: Qualitative Error Analysis on Dev Set

**LaTeX format:**

```
\begin{tabular}{|p{3cm}|p{4cm}|p{5cm}|p{2cm}|}
\hline
Error Type & Example of Error & Error Description & Statistics \\
\hline
\textbf{Missing or Incorrect Condition} & 
NL: "please give me the flight times i would like to fly from boston to baltimore in the morning before 8" \\
GT: \texttt{SELECT DISTINCT flight\_1.departure\_time FROM flight flight\_1, ... WHERE ... AND flight\_1.departure\_time < 800} \\
Pred: \texttt{SELECT DISTINCT flight\_1.arrival\_time, flight\_1.departure\_time FROM flight flight\_1, ... WHERE flight\_1.departure\_time BETWEEN 0 AND 1200 AND ... AND flight\_1.departure\_time  800} &
The model generates incorrect time conditions. In the example, the ground truth uses \texttt{departure\_time < 800} to represent "before 8 AM", but the model generates \texttt{departure\_time BETWEEN 0 AND 1200} (incorrect time range) and also has a syntax error (\texttt{flight\_1.departure\_time  800} missing operator). The model struggles with temporal constraints, often generating incorrect time comparisons or missing specific time conditions mentioned in natural language queries. &
69/84 (82.1\% of incorrect predictions) \\
\hline
\textbf{Wrong Column Selection} & 
NL: "please give me the flight times i would like to fly from boston to baltimore in the morning before 8" \\
GT: \texttt{SELECT DISTINCT flight\_1.departure\_time FROM flight ...} \\
Pred: \texttt{SELECT DISTINCT flight\_1.arrival\_time, flight\_1.departure\_time FROM flight ...} \\
NL: "what does iah mean" \\
GT: \texttt{SELECT DISTINCT airport\_1.airport\_code FROM airport airport\_1 WHERE airport\_1.airport\_code = 'IAH'} \\
Pred: \texttt{SELECT DISTINCT airline\_1.airline\_code FROM airline airline\_1 WHERE airline\_1.airline\_code = 'IAH'} &
The model selects incorrect columns or tables. In the first example, it adds \texttt{arrival\_time} when only \texttt{departure\_time} is requested. In the second example, it queries the \texttt{airline} table instead of the \texttt{airport} table, selecting \texttt{airline\_code} instead of \texttt{airport\_code}. This error occurs when the model misinterprets which entity or attribute is being queried, often confusing similar concepts (e.g., airports vs. airlines, arrival vs. departure times). &
15/84 (17.9\% of incorrect predictions) \\
\hline
\textbf{Wrong Aggregation or Subquery Structure} & 
NL: "list all flights from boston to san francisco with the maximum number of stops" \\
GT: \texttt{SELECT DISTINCT flight\_1.flight\_id FROM flight flight\_1, ... WHERE ... AND flight\_1.stops = (SELECT MAX(stops) FROM flight flight\_1, ... WHERE ...)} \\
Pred: \texttt{SELECT DISTINCT flight\_1.flight\_id FROM flight flight\_1, ... WHERE ... AND flight\_1.stops = (SELECT MAX(flight\_1.stops >= 1000))} &
The model generates malformed subqueries or incorrect aggregation logic. In the example, the ground truth correctly uses a subquery with \texttt{MAX(stops)} to find flights with maximum stops, but the model generates an invalid condition \texttt{MAX(flight\_1.stops >= 1000)} which is syntactically incorrect. The model struggles with complex aggregation queries, especially those requiring subqueries to find maximum or minimum values, often generating invalid SQL syntax. &
1/84 (1.2\% of incorrect predictions) \\
\hline
\end{tabular}
```

**Plain text format:**

| Error Type | Example of Error | Error Description | Statistics |
|------------|------------------|-------------------|------------|
| **Missing or Incorrect Condition** | NL: "please give me the flight times i would like to fly from boston to baltimore in the morning before 8"<br><br>GT: `SELECT DISTINCT flight_1.departure_time FROM flight flight_1, ... WHERE ... AND flight_1.departure_time < 800`<br><br>Pred: `SELECT DISTINCT flight_1.arrival_time, flight_1.departure_time FROM flight flight_1, ... WHERE flight_1.departure_time BETWEEN 0 AND 1200 AND ... AND flight_1.departure_time  800` | The model generates incorrect time conditions. In the example, the ground truth uses `departure_time < 800` to represent "before 8 AM", but the model generates `departure_time BETWEEN 0 AND 1200` (incorrect time range) and also has a syntax error (`flight_1.departure_time  800` missing operator). The model struggles with temporal constraints, often generating incorrect time comparisons or missing specific time conditions mentioned in natural language queries. | 69/84 (82.1% of incorrect predictions) |
| **Wrong Column Selection** | NL: "please give me the flight times i would like to fly from boston to baltimore in the morning before 8"<br><br>GT: `SELECT DISTINCT flight_1.departure_time FROM flight ...`<br><br>Pred: `SELECT DISTINCT flight_1.arrival_time, flight_1.departure_time FROM flight ...`<br><br>NL: "what does iah mean"<br><br>GT: `SELECT DISTINCT airport_1.airport_code FROM airport airport_1 WHERE airport_1.airport_code = 'IAH'`<br><br>Pred: `SELECT DISTINCT airline_1.airline_code FROM airline airline_1 WHERE airline_1.airline_code = 'IAH'` | The model selects incorrect columns or tables. In the first example, it adds `arrival_time` when only `departure_time` is requested. In the second example, it queries the `airline` table instead of the `airport` table, selecting `airline_code` instead of `airport_code`. This error occurs when the model misinterprets which entity or attribute is being queried, often confusing similar concepts (e.g., airports vs. airlines, arrival vs. departure times). | 15/84 (17.9% of incorrect predictions) |
| **Wrong Aggregation or Subquery Structure** | NL: "list all flights from boston to san francisco with the maximum number of stops"<br><br>GT: `SELECT DISTINCT flight_1.flight_id FROM flight flight_1, ... WHERE ... AND flight_1.stops = (SELECT MAX(stops) FROM flight flight_1, ... WHERE ...)`<br><br>Pred: `SELECT DISTINCT flight_1.flight_id FROM flight flight_1, ... WHERE ... AND flight_1.stops = (SELECT MAX(flight_1.stops >= 1000))` | The model generates malformed subqueries or incorrect aggregation logic. In the example, the ground truth correctly uses a subquery with `MAX(stops)` to find flights with maximum stops, but the model generates an invalid condition `MAX(flight_1.stops >= 1000)` which is syntactically incorrect. The model struggles with complex aggregation queries, especially those requiring subqueries to find maximum or minimum values, often generating invalid SQL syntax. | 1/84 (1.2% of incorrect predictions) |

**Total incorrect predictions**: 84/466 (18.0%)  
**Total correct predictions**: 382/466 (82.0%)

