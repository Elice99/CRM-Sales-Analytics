SELECT *
FROM [dbo].[accounts]
 
 --Capitalize sector consistently
UPDATE [dbo].[accounts]
SET [sector] = UPPER(LEFT([sector],1)) + LOWER(SUBSTRING([sector], 2, LEN([sector])));

--Drop subsidiary column since its have 80% Null
ALTER TABLE [dbo].[accounts]
DROP COLUMN [subsidiary_of];

--Creating a new account_tier column to categorize business account by employees
ALTER TABLE [dbo].[accounts]
ADD account_tier VARCHAR(20);

--Categorizing account into tiers based on employees
UPDATE [dbo].[accounts]
SET [account_tier] = CASE
						WHEN [employees] BETWEEN 1 AND 9 THEN 'Micro_Enterprise'
						WHEN [employees] BETWEEN 10 AND 49 THEN 'Small_Enterprise'
						WHEN [employees] BETWEEN 50 AND 249 THEN 'Medium_Enterprise'
						ELSE 'Large_Enterprise'
					 END;

