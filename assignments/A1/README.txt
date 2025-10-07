How to run:
- run each script using CLI
- python ahmem73_part1.py 
- python ahmem73_part2.py 

AI Use:
I used chatgpt-5 to help me understand the class provided code. furthermore, I also used gpt as 
a coding assistant for syntax related or graphing related help. the prompts:

- by_year = (data[data['Year']==2018]).drop(columns=["World regions according to OWID","Code"]) what does this do
-#create np.array for gdp and happiness where happiness score is above 4.5 happiness=[] gdp=[] for row in df.iterrows(): print(row, "\n\n\n") if row[1]['Cantril ladder score']>4.5: happiness.append(row[1]['Cantril ladder score']) gdp.append(row[1]['GDP per capita, PPP (constant 2021 international $)']) explain what this snippet does
- <provided the linear regression class> explain each function of this class to me 
- X_ = X[...,1].ravel() what does this code do next

- I have a list of tuples with (lr, epoch, beta) that contain Gradient descent values for each run. how Can 
i plot all of these lines on the same plot. make sure each line has a different color and their labeled showing the 
epoch value and learning rate
- I have a list of tuples with (lr, epoch, beta) that contain Gradient descent values for each run. i also have 
the OLS solution (beta_ols). beta_ols and beta are both np arrays. how can i pick which tuple of (lr, epoch and beta)
have the closest coeffecients to the beta_ols solution. 

CO2 usage: 4.32g * (query) = 4.32(6) = 25.92g