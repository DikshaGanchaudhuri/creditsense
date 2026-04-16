-- How many good vs bad credit risks?
SELECT class, COUNT(*) AS count
FROM credit_data
GROUP BY class;

-- Average loan amount and duration by risk class
SELECT
    class,
    ROUND(AVG(credit_amount)) AS avg_loan_amount,
    ROUND(AVG(duration))      AS avg_duration_months
FROM credit_data
GROUP BY class;

-- Age distribution across risk classes
SELECT
    class,
    MIN(age) AS youngest,
    ROUND(AVG(age)) AS avg_age,
    MAX(age) AS oldest
FROM credit_data
GROUP BY class;

-- Most common loan purposes
SELECT purpose, COUNT(*) AS count
FROM credit_data
GROUP BY purpose
ORDER BY count DESC;

-- Default rate by housing type
SELECT
    housing,
    COUNT(*) AS total,
    SUM(CASE WHEN class = 'bad' THEN 1 ELSE 0 END) AS defaults,
    ROUND(100.0 * SUM(CASE WHEN class = 'bad' THEN 1 ELSE 0 END) / COUNT(*), 1) AS default_rate_pct
FROM credit_data
GROUP BY housing
ORDER BY default_rate_pct DESC;