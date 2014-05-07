1. What customers are from the UK
SELECT * FROM Customers where country = "UK";

2. What is the name of the customer who has the most orders?
SELECT customerID, count(*) FROM orders group by customerID order by count(*) desc 

3. What supplier has the highest average product price?
SELECT SupplierID, avg(Price) FROM [Products] group by supplierID order by avg(Price) desc limit 1

4. What category has the most orders?
SELECT categoryid, count(categoryid)
FROM OrderDetails JOIN Products ON OrderDetails.ProductID = Products.ProductID
group by categoryid
order by count(categoryid) desc
limit 1

5. What employee made the most sales (by number of sales)?

SELECT employeeID, count(employeeID) FROM [Orders] group by employeeID order by count(employeeID) desc

6. What employee made the most sales (by value of sales)?


7. What employees have BS degrees? (Hint: Look at LIKE operator)


8. What supplier has the highest average product price *assuming they have at least 2 products* (Hint: Look at the HAVING operator)

