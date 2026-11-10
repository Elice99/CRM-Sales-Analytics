CREATE DATABASE CRM_Sales_Opportunity

USE [CRM_Sales_Opportunity]

SELECT *
FROM [dbo].[sales_pipeline]


--Creating a new is_active column to track early stage deals
ALTER TABLE [dbo].[sales_pipeline]
ADD is_active BIT;


--Flag early stage deals as is_active (Engaging, lead, prospect)
UPDATE [dbo].[sales_pipeline]
SET is_active = CASE
					WHEN deal_stage IN ('Engaging', 'Lead', 'Prospect') THEN 1
					ELSE 0
				END;


-- Fix Null in close_value for deals not yet Won (set to 0)
UPDATE [dbo].[sales_pipeline]
SET [close_value] = 0
WHERE close_value IS NULL AND deal_stage != 'Won';


-- Fix Null close_date for closed deals either won or lost set as currrent date
UPDATE [dbo].[sales_pipeline]
SET [close_date] = GETDATE()
WHERE [close_date] IS NULL AND deal_stage IN ('Won', 'Lost');


--Handling Null account(Primary Key) set as Unknown
--If there is a Null Acct it will probably be from the active deal, since the deal is till ongoing
UPDATE [dbo].[sales_pipeline]
SET [account] = 'Unknown'
WHERE [account] IS NULL ;


--Capitalize deal_stage consistently
UPDATE [dbo].[sales_pipeline]
SET [deal_stage] = UPPER(LEFT([deal_stage],1)) + LOWER(SUBSTRING([deal_stage], 2, LEN([deal_stage])));


--Removing any duplicate opportunities(keep the earliest engaging_date) if any
WITH Duplicates_opp AS (
	SELECT [opportunity_id],
		ROW_NUMBER() OVER(PARTITION BY [opportunity_id] ORDER BY [engage_date] ASC) rn
	FROM [dbo].[sales_pipeline]
)
DELETE FROM [dbo].[sales_pipeline]
WHERE [opportunity_id] IN (
	SELECT [opportunity_id] 
	FROM Duplicates_opp
	WHERE rn > 1
);


-- Audit summary by deal_stage
SELECT 
	[deal_stage],
	COUNT(*) total_opportunities,
	SUM(CASE WHEN [account] IS NULL THEN 1 ELSE 0 END) null_acct,
	SUM(CASE WHEN [close_date] IS NULL THEN 1 ELSE 0 END) null_close_date,
	SUM(CASE WHEN [close_value] IS NULL THEN 1 ELSE 0 END) null_close_values
FROM [dbo].[sales_pipeline]
GROUP BY [deal_stage]
ORDER BY total_opportunities DESC

SELECT *
FROM [dbo].[accounts]