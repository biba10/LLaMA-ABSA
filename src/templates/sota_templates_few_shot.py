FEW_SHOT_ACOS_LAPTOP16 = """
Input: \"\"\"acer wants $ 170 to just look at it then add the repair cost on top of that .\"\"\"
Sentiment elements: [("acer", "support price", "neutral", "null")]

Input: \"\"\"update : i repaired it myself for $ 12 .\"\"\"
Sentiment elements: [("null", "laptop general", "neutral", "null")]

Input: \"\"\"i had nothing to lose since it was a paper weight otherwise .\"\"\"
Sentiment elements: [("null", "laptop general", "negative", "null")]

Input: \"\"\"the shame of it is knowing it took me 15 minutes and $ 12 to fix it and acer wanted to rob me of $ 170 just to look at it .\"\"\"
Sentiment elements: [("acer", "support general", "negative", "null")]

Input: \"\"\"first one that they shipped was obviously defective , super slow and speakers were garbled .\"\"\"
Sentiment elements: [("null", "shipping general", "negative", "defective"), ("null", "shipping general", "negative", "slow"), ("speakers", "multimedia_devices general", "negative", "garbled")]

Input: \"\"\"the replacement i got was much better , but still too slow for my expectations .\"\"\"
Sentiment elements: [("replacement", "laptop quality", "negative", "slow")]

Input: \"\"\"i wound up returning it .\"\"\"
Sentiment elements: [("null", "laptop general", "negative", "null")]

Input: \"\"\"this works fine for that .\"\"\"
Sentiment elements: [("null", "laptop quality", "positive", "fine")]

Input: \"\"\"october 12 , 2017 - - started having trouble maintaining connection to wifi ( spectrum service ) , but usually after several loops re - entering password , connection would be re - established .\"\"\"
Sentiment elements: [("wifi", "ports quality", "negative", "trouble")]

Input: \"\"\"sometimes had to do several times , but thought it might be an idiosyncrady of this model .\"\"\"
Sentiment elements: [("model", "hardware general", "negative", "null")]

"""

FEW_SHOT_ACOS_REST16 = """
Input: \"\"\"judging from previous posts this used to be a good place , but not any longer .\"\"\"
Sentiment elements: [("place", "restaurant general", "negative", "not any longer")]

Input: \"\"\"we , there were four of us , arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude .\"\"\"
Sentiment elements: [("staff", "service general", "negative", "rude")]

Input: \"\"\"they never brought us complimentary noodles , ignored repeated requests for sugar , and threw our dishes on the table .\"\"\"
Sentiment elements: [("null", "service general", "negative", "null")]

Input: \"\"\"the food was lousy - too sweet or too salty and the portions tiny .\"\"\"
Sentiment elements: [("food", "food quality", "negative", "lousy"), ("food", "food quality", "negative", "too sweet"), ("food", "food quality", "negative", "too salty"), ("portions", "food style_options", "negative", "tiny")]

Input: \"\"\"after all that , they complained to me about the small tip .\"\"\"
Sentiment elements: [("null", "service general", "negative", "complained")]

Input: \"\"\"avoid this place !\"\"\"
Sentiment elements: [("place", "restaurant general", "negative", "avoid")]

Input: \"\"\"i have eaten at saul , many times , the food is always consistently , outrageously good .\"\"\"
Sentiment elements: [("food", "food quality", "positive", "outrageously good")]

Input: \"\"\"saul is the best restaurant on smith street and in brooklyn .\"\"\"
Sentiment elements: [("saul", "restaurant general", "positive", "best")]

Input: \"\"\"the duck confit is always amazing and the foie gras terrine with figs was out of this world .\"\"\"
Sentiment elements: [("foie gras terrine with figs", "food quality", "positive", "out of this world"), ("duck confit", "food quality", "positive", "amazing")]

Input: \"\"\"the wine list is interesting and has many good values .\"\"\"
Sentiment elements: [("wine list", "drinks style_options", "positive", "interesting"), ("wine list", "drinks prices", "positive", "good values")]

"""

FEW_SHOT_ASQP_REST15 = """
Input: \"\"\"The wait here is long for dim sum , but if you do n"t like sharing tables or if the typical raucous dim sum atmosphere is not your gig , this is a sleek ( for Chinatown ) alternative .\"\"\"
Sentiment elements: [("wait", "service general", "negative", "long"), ("atmosphere", "ambience general", "negative", "raucous"), ("null", "restaurant miscellaneous", "negative", "sleek")]

Input: \"\"\"Just because it "s cheap does NOT mean the portions are small or the food is nasty , IT IS GREAT !\"\"\"
Sentiment elements: [("food", "food quality", "positive", "GREAT"), ("null", "restaurant prices", "positive", "cheap")]

Input: \"\"\"Food is excellent .\"\"\"
Sentiment elements: [("Food", "food quality", "positive", "excellent")]

Input: \"\"\"As always we had a great glass of wine while we waited .\"\"\"
Sentiment elements: [("glass of wine", "drinks quality", "positive", "great")]

Input: \"\"\"I can not imagine a friendlier staff working in a restaurant .\"\"\"
Sentiment elements: [("staff", "service general", "positive", "friendlier")]

Input: \"\"\"Also , specify if you like your food spicy- its rather bland if you do n"t .\"\"\"
Sentiment elements: [("food", "food quality", "negative", "bland")]

Input: \"\"\"Big Wong gets big Ups for a fine establishment .\"\"\"
Sentiment elements: [("Big Wong", "restaurant general", "positive", "big Ups"), ("Big Wong", "restaurant general", "positive", "fine")]

Input: \"\"\"I was pleasantly suprised .\"\"\"
Sentiment elements: [("null", "restaurant general", "positive", "pleasantly suprised")]

Input: \"\"\"We all agreed that mare is one of the best seafood restaurants in New York .\"\"\"
Sentiment elements: [("mare", "restaurant general", "positive", "best")]

Input: \"\"\"I ca n"t wait to go back .\"\"\"
Sentiment elements: [("null", "restaurant general", "positive", "go back")]

"""

FEW_SHOT_ASQP_REST16 = """
Input: \"\"\"We have gone for dinner only a few times but the same great quality and service is given .\"\"\"
Sentiment elements: [("service", "service general", "positive", "great"), ("dinner", "food quality", "positive", "great quality")]

Input: \"\"\"Its dark , and cozy . . there is always jazz music playing when we go .\"\"\"
Sentiment elements: [("null", "ambience general", "positive", "cozy")]

Input: \"\"\"This place has great indian chinese food .\"\"\"
Sentiment elements: [("indian chinese food", "food quality", "positive", "great")]

Input: \"\"\"Not what I would expect for the price and prestige of this location .\"\"\"
Sentiment elements: [("location", "restaurant prices", "neutral", "expect"), ("location", "restaurant miscellaneous", "neutral", "expect")]

Input: \"\"\"Finally a reliable Chinese restaurant !\"\"\"
Sentiment elements: [("Chinese restaurant", "restaurant general", "positive", "reliable")]

Input: \"\"\"The lobster knuckles ( special of the day ) were ok , but pretty tasteless .\"\"\"
Sentiment elements: [("lobster knuckles", "food style_options", "neutral", "ok"), ("lobster knuckles", "food quality", "negative", "tasteless")]

Input: \"\"\"The menu is fairly simple without much descriptions .\"\"\"
Sentiment elements: [("menu", "food style_options", "neutral", "simple")]

Input: \"\"\"WORST PLACE ON SMITH STREET IN BROOKLYN\"\"\"
Sentiment elements: [("PLACE", "restaurant general", "negative", "WORST")]

Input: \"\"\"The staff has been nice , but they seemed really stressed and the unisex bathroom needs to be cleaned more often .\"\"\"
Sentiment elements: [("staff", "service general", "positive", "nice"), ("staff", "service general", "negative", "stressed"), ("unisex bathroom", "ambience general", "negative", "needs to be cleaned")]

Input: \"\"\"I absolutely love this place ! ! !\"\"\"
Sentiment elements: [("place", "restaurant general", "positive", "love")]

"""

FEW_SHOT_ASTE_LAPTOP14 = """
Input: \"\"\"I charge it at night and skip taking the cord with me because of the good battery life .\"\"\"
Sentiment elements: [("battery life", "good", "positive")]

Input: \"\"\"it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous .\"\"\"
Sentiment elements: [("quality", "high", "positive"), ("GUI", "killer", "positive"), ("applications", "good", "positive"), ("use", "easy", "positive")]

Input: \"\"\"Easy to start up and does not overheat as much as other laptops .\"\"\"
Sentiment elements: [("start up", "Easy", "positive")]

Input: \"\"\"Great laptop that offers many great features !\"\"\"
Sentiment elements: [("features", "great", "positive")]

Input: \"\"\"One night I turned the freaking thing off after using it , the next day I turn it on , no GUI , screen all dark , power light steady , hard drive light steady and not flashing as it usually does .\"\"\"
Sentiment elements: [("GUI", "no", "negative"), ("screen", "dark", "negative"), ("power light", "steady", "neutral"), ("hard drive light", "steady", "negative")]

Input: \"\"\"However , the multi-touch gestures and large tracking area make having an external mouse unnecessary ( unless you "re gaming ) .\"\"\"
Sentiment elements: [("external mouse", "unnecessary", "neutral")]

Input: \"\"\"I love the way the entire suite of software works together .\"\"\"
Sentiment elements: [("suite of software", "love", "positive")]

Input: \"\"\"The speed is incredible and I am more than satisfied .\"\"\"
Sentiment elements: [("speed", "incredible", "positive"), ("speed", "satisfied", "positive")]

Input: \"\"\"I can barely use any usb devices because they will not stay connected properly .\"\"\"
Sentiment elements: [("usb devices", "not stay connected properly", "negative")]

Input: \"\"\"When I finally had everything running with all my software installed I plugged in my droid to recharge and the system crashed .\"\"\"
Sentiment elements: [("system", "crashed", "negative")]

"""

FEW_SHOT_ASTE_REST4 = """
Input: \"\"\"But the staff was so horrible to us .\"\"\"
Sentiment elements: [("staff", "horrible", "negative")]

Input: \"\"\"To be completely fair , the only redeeming factor was the food , which was above average , but could n"t make up for all the other deficiencies of Teodora .\"\"\"
Sentiment elements: [("food", "above average", "positive")]

Input: \"\"\"The food is uniformly exceptional , with a very capable kitchen which will proudly whip up whatever you feel like eating , whether it "s on the menu or not .\"\"\"
Sentiment elements: [("food", "exceptional", "positive"), ("kitchen", "capable", "positive")]

Input: \"\"\"Our agreed favorite is the orrechiete with sausage and chicken ( usually the waiters are kind enough to split the dish in half so you get to sample both meats ) .\"\"\"
Sentiment elements: [("orrechiete with sausage and chicken", "favorite", "positive"), ("waiters", "kind", "positive")]

Input: \"\"\"The Bagels have an outstanding taste with a terrific texture , both chewy yet not gummy .\"\"\"
Sentiment elements: [("Bagels", "outstanding", "positive"), ("Bagels", "terrific", "positive"), ("Bagels", "chewy", "positive"), ("Bagels", "gummy", "positive")]

Input: \"\"\"Nevertheless the food itself is pretty good .\"\"\"
Sentiment elements: [("food", "good", "positive")]

Input: \"\"\"They did not have mayonnaise , forgot our toast , left out ingredients ( ie cheese in an omelet ) , below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it .\"\"\"
Sentiment elements: [("toast", "forgot", "negative"), ("bacon", "over cooked", "negative"), ("cheese", "left out", "neutral"), ("ingredients", "left out", "negative"), ("plate", "over cooked", "neutral"), ("omelet", "left out", "neutral")]

Input: \"\"\"The design and atmosphere is just as good .\"\"\"
Sentiment elements: [("design", "good", "positive"), ("atmosphere", "good", "positive")]

Input: \"\"\"The seats are uncomfortable if you are sitting against the wall on wooden benches .\"\"\"
Sentiment elements: [("seats", "uncomfortable", "negative")]

Input: \"\"\"My suggestion is to eat family style because you "ll want to try the other dishes .\"\"\"
Sentiment elements: [("eat family style", "suggestion", "positive")]

"""

FEW_SHOT_ASTE_REST15 = """
Input: \"\"\"Judging from previous posts this used to be a good place , but not any longer .\"\"\"
Sentiment elements: [("place", "good", "negative")]

Input: \"\"\"We , there were four of us , arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude .\"\"\"
Sentiment elements: [("staff", "rude", "negative")]

Input: \"\"\"The food was lousy - too sweet or too salty and the portions tiny .\"\"\"
Sentiment elements: [("food", "lousy", "negative"), ("food", "too sweet", "negative"), ("food", "too salty", "negative"), ("portions", "tiny", "negative")]

Input: \"\"\"Avoid this place !\"\"\"
Sentiment elements: [("place", "Avoid", "negative")]

Input: \"\"\"I have eaten at Saul , many times , the food is always consistently , outrageously good .\"\"\"
Sentiment elements: [("food", "good", "positive")]

Input: \"\"\"Saul is the best restaurant on Smith Street and in Brooklyn .\"\"\"
Sentiment elements: [("Saul", "best", "positive")]

Input: \"\"\"The duck confit is always amazing and the foie gras terrine with figs was out of this world .\"\"\"
Sentiment elements: [("foie gras terrine with figs", "out of this world", "positive"), ("duck confit", "amazing", "positive")]

Input: \"\"\"The wine list is interesting and has many good values .\"\"\"
Sentiment elements: [("wine list", "interesting", "positive"), ("wine list", "good values", "positive")]

Input: \"\"\"I was very disappointed with this restaurant .\"\"\"
Sentiment elements: [("restaurant", "disappointed", "negative")]

Input: \"\"\"Food was okay , nothing great .\"\"\"
Sentiment elements: [("Food", "okay", "neutral"), ("Food", "nothing great", "neutral")]

"""

FEW_SHOT_ASTE_REST16 = """
Input: \"\"\"Judging from previous posts this used to be a good place , but not any longer .\"\"\"
Sentiment elements: [("place", "good", "negative")]

Input: \"\"\"We , there were four of us , arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude .\"\"\"
Sentiment elements: [("staff", "rude", "negative")]

Input: \"\"\"The food was lousy - too sweet or too salty and the portions tiny .\"\"\"
Sentiment elements: [("food", "lousy", "negative"), ("food", "too sweet", "negative"), ("food", "too salty", "negative"), ("portions", "tiny", "negative")]

Input: \"\"\"Avoid this place !\"\"\"
Sentiment elements: [("place", "Avoid", "negative")]

Input: \"\"\"I have eaten at Saul , many times , the food is always consistently , outrageously good .\"\"\"
Sentiment elements: [("food", "good", "positive")]

Input: \"\"\"Saul is the best restaurant on Smith Street and in Brooklyn .\"\"\"
Sentiment elements: [("Saul", "best", "positive")]

Input: \"\"\"The duck confit is always amazing and the foie gras terrine with figs was out of this world .\"\"\"
Sentiment elements: [("foie gras terrine with figs", "out of this world", "positive"), ("duck confit", "amazing", "positive")]

Input: \"\"\"The wine list is interesting and has many good values .\"\"\"
Sentiment elements: [("wine list", "interesting", "positive"), ("wine list", "good values", "positive")]

Input: \"\"\"I was very disappointed with this restaurant .\"\"\"
Sentiment elements: [("restaurant", "disappointed", "negative")]

Input: \"\"\"Food was okay , nothing great .\"\"\"
Sentiment elements: [("Food", "okay", "neutral"), ("Food", "nothing great", "neutral")]

"""

FEW_SHOT_TASD_REST15 = """
Input: \"\"\"Judging from previous posts this used to be a good place , but not any longer .\"\"\"
Sentiment elements: [("place", "restaurant general", "negative")]

Input: \"\"\"We , there were four of us , arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude .\"\"\"
Sentiment elements: [("staff", "service general", "negative")]

Input: \"\"\"They never brought us complimentary noodles , ignored repeated requests for sugar , and threw our dishes on the table .\"\"\"
Sentiment elements: [("null", "service general", "negative")]

Input: \"\"\"The food was lousy - too sweet or too salty and the portions tiny .\"\"\"
Sentiment elements: [("food", "food quality", "negative"), ("portions", "food style_options", "negative")]

Input: \"\"\"After all that , they complained to me about the small tip .\"\"\"
Sentiment elements: [("null", "service general", "negative")]

Input: \"\"\"Avoid this place !\"\"\"
Sentiment elements: [("place", "restaurant general", "negative")]

Input: \"\"\"I have eaten at Saul , many times , the food is always consistently , outrageously good .\"\"\"
Sentiment elements: [("food", "food quality", "positive")]

Input: \"\"\"Saul is the best restaurant on Smith Street and in Brooklyn .\"\"\"
Sentiment elements: [("Saul", "restaurant general", "positive")]

Input: \"\"\"The duck confit is always amazing and the foie gras terrine with figs was out of this world .\"\"\"
Sentiment elements: [("foie gras terrine with figs", "food quality", "positive"), ("duck confit", "food quality", "positive")]

Input: \"\"\"The wine list is interesting and has many good values .\"\"\"
Sentiment elements: [("wine list", "drinks style_options", "positive"), ("wine list", "drinks prices", "positive")]

"""

FEW_SHOT_TASD_REST15 = """
Input: \"\"\"Judging from previous posts this used to be a good place , but not any longer .\"\"\"
Sentiment elements: [("place", "restaurant general", "negative")]

Input: \"\"\"We , there were four of us , arrived at noon - the place was empty - and the staff acted like we were imposing on them and they were very rude .\"\"\"
Sentiment elements: [("staff", "service general", "negative")]

Input: \"\"\"They never brought us complimentary noodles , ignored repeated requests for sugar , and threw our dishes on the table .\"\"\"
Sentiment elements: [("null", "service general", "negative")]

Input: \"\"\"The food was lousy - too sweet or too salty and the portions tiny .\"\"\"
Sentiment elements: [("food", "food quality", "negative"), ("portions", "food style_options", "negative")]

Input: \"\"\"After all that , they complained to me about the small tip .\"\"\"
Sentiment elements: [("null", "service general", "negative")]

Input: \"\"\"Avoid this place !\"\"\"
Sentiment elements: [("place", "restaurant general", "negative")]

Input: \"\"\"I have eaten at Saul , many times , the food is always consistently , outrageously good .\"\"\"
Sentiment elements: [("food", "food quality", "positive")]

Input: \"\"\"Saul is the best restaurant on Smith Street and in Brooklyn .\"\"\"
Sentiment elements: [("Saul", "restaurant general", "positive")]

Input: \"\"\"The duck confit is always amazing and the foie gras terrine with figs was out of this world .\"\"\"
Sentiment elements: [("foie gras terrine with figs", "food quality", "positive"), ("duck confit", "food quality", "positive")]

Input: \"\"\"The wine list is interesting and has many good values .\"\"\"
Sentiment elements: [("wine list", "drinks style_options", "positive"), ("wine list", "drinks prices", "positive")]

"""

FEW_SHOT_PROMPTS_SOTA = {
    "acos/laptop16": FEW_SHOT_ACOS_LAPTOP16,
    "acos/rest16": FEW_SHOT_ACOS_REST16,
    "asqp/rest15": FEW_SHOT_ASQP_REST15,
    "asqp/rest16": FEW_SHOT_ASQP_REST16,
    "aste/laptop14": FEW_SHOT_ASTE_LAPTOP14,
    "aste/rest14": FEW_SHOT_ASTE_REST4,
    "aste/rest15": FEW_SHOT_ASTE_REST15,
    "aste/rest16": FEW_SHOT_ASTE_REST16,
    "tasd/rest15": FEW_SHOT_TASD_REST15,
    "tasd/rest16": FEW_SHOT_TASD_REST15
}
