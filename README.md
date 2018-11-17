# Autocomplete-NLP
Sentense word nlp autocomplete

**Imagine** that you were a representative replying to customer online and you are asking more or less the same questions over and over to your customer. Would you like to get automatic suggestions instead of typing the same thing again and again ?

An **autocomplete** can be helpful, faster, convenient and also correct any grammatical / spelling error at the same time.


**_Project_**:

In the jupyter notebook in this project, we select an history of sentenses written by the representatives and the customer, format and correct them using a few regex rules and count them so we can estimate their frequency and likelyness to be useful again.
After the calculation of a similarity matrix based on the sklearn **tfidf** tool (frequency and normalization of words), we use this matrix to calculate the similarity between the new few words written by the representative and the history of messages written in the past.
The Autocomplete will recognize the closest sentenses and rank 3 final proposals:

If you were to type: `What is your`,
the tool would suggest:
> What is your account number?,
 What is your order number?,
 What is your phone number?
 
 
if you were to type: `Let me`
the tool would suggest:
> Let me investigate, Let me assist you, Let me look
 
 if you were to type without uppercase: `when was`
 > When was the last time you changed your password?,
 When was your flight scheduled for?,
 When was the last time you tried?
 
  
  
   
 **_Improvements_**:
 1. clean up the "Mr. Smith" and "Ms. Smith" in the dataset
 2. Match the letter to the words (spelling match) and then match the words to the representative sentenses history.
 3. Build an evaluation of the results:
 - a. offline: using unseen conversations between the representative and the customer, input the prefix of the representative in the model and match to see if the actual representative sentense is part of the 3 ranked proposals.
 - b. online: count the number of time the representative actually select a proposals and count the number of time the representative decides to ignore them.
 4. improve the system by first matching the customer sentenses to a topic id context in order to better predict the representative answers   
 
    

