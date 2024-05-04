RESTAURANT_CATEGORIES = '"ambience general", "drinks prices", "drinks quality", "drinks style_options", "food general", "food prices", "food quality", "food style_options", "location general", "restaurant general", "restaurant miscellaneous", "restaurant prices", "service general"'

RESTAURANT_CATEGORIES_TASD_REST16 = '"ambience general", "drinks prices", "drinks quality", "drinks style_options", "food prices", "food quality", "food style_options", "location general", "restaurant general", "restaurant miscellaneous", "restaurant prices", "service general"'

LAPTOP_CATEGORIES = '"battery design_features", "battery general", "battery operation_performance", "battery quality", "company design_features", "company general", "company operation_performance", "company price", "company quality", "cpu design_features", "cpu general", "cpu operation_performance", "cpu price", "cpu quality", "display design_features", "display general", "display operation_performance", "display price", "display quality", "display usability", "fans&cooling design_features", "fans&cooling general", "fans&cooling operation_performance", "fans&cooling quality", "graphics design_features", "graphics general", "graphics operation_performance", "graphics usability", "hard_disc design_features", "hard_disc general", "hard_disc miscellaneous", "hard_disc operation_performance", "hard_disc price", "hard_disc quality", "hard_disc usability", "hardware design_features", "hardware general", "hardware operation_performance", "hardware price", "hardware quality", "hardware usability", "keyboard design_features", "keyboard general", "keyboard miscellaneous", "keyboard operation_performance", "keyboard portability", "keyboard price", "keyboard quality", "keyboard usability", "laptop connectivity", "laptop design_features", "laptop general", "laptop miscellaneous", "laptop operation_performance", "laptop portability", "laptop price", "laptop quality", "laptop usability", "memory design_features", "memory general", "memory operation_performance", "memory quality", "memory usability", "motherboard general", "motherboard operation_performance", "motherboard quality", "mouse design_features", "mouse general", "mouse usability", "multimedia_devices connectivity", "multimedia_devices design_features", "multimedia_devices general", "multimedia_devices operation_performance", "multimedia_devices price", "multimedia_devices quality", "multimedia_devices usability", "optical_drives design_features", "optical_drives general", "optical_drives operation_performance", "optical_drives usability", "os design_features", "os general", "os miscellaneous", "os operation_performance", "os price", "os quality", "os usability", "out_of_scope design_features", "out_of_scope general", "out_of_scope operation_performance", "out_of_scope usability", "ports connectivity", "ports design_features", "ports general", "ports operation_performance", "ports portability", "ports quality", "ports usability", "power_supply connectivity", "power_supply design_features", "power_supply general", "power_supply operation_performance", "power_supply quality", "shipping general", "shipping operation_performance", "shipping price", "shipping quality", "software design_features", "software general", "software operation_performance", "software portability", "software price", "software quality", "software usability", "support design_features", "support general", "support operation_performance", "support price", "support quality", "warranty general", "warranty quality"'

BASIC_PROMPT_TASD = """According to the following sentiment elements definition:

- The "aspect term" refers to a specific feature, attribute, or aspect of a product or service on which a user can express an opinion. Explicit aspect terms appear explicitly as a substring of the given text. The aspect term might be "null" for the implicit aspect.

- The "aspect category" refers to the category that aspect belongs to, and the available categories include: {categories}.

- The "sentiment polarity" refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Triplets with objective sentiment polarity should be ignored.

Please carefully follow the instructions. Ensure that aspect terms are recognized as exact matches in the review or are "null" for implicit aspects. Ensure that aspect categories are from the available categories. Ensure that sentiment polarities are from the available polarities.

Recognize all sentiment elements with their corresponding aspect terms, aspect categories, and sentiment polarity in the given input text (review). Provide your response in the format of a Python list of tuples: 'Sentiment elements: [("aspect term", "aspect category", "sentiment polarity"), ...]'. Note that ", ..." indicates that there might be more tuples in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.

"""

BASIC_PROMPT_QUADRUPLET_TASKS = """According to the following sentiment elements definition:

- The "aspect term" refers to a specific feature, attribute, or aspect of a product or service on which a user can express an opinion. Explicit aspect terms appear explicitly as a substring of the given text. The aspect term might be "null" for the implicit aspect.

- The "aspect category" refers to the category that aspect belongs to, and the available categories include: {categories}.

- The "sentiment polarity" refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Quadruplets with objective sentiment polarity should be ignored.

- The "opinion term" refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service. Explicit opinion terms appear explicitly as a substring of the given text. The opinion term might be "null" for the implicit opinion.

Please carefully follow the instructions. Ensure that aspect terms are recognized as exact matches in the review or are "null" for implicit aspects. Ensure that aspect categories are from the available categories. Ensure that sentiment polarities are from the available polarities. Ensure that opinion terms are recognized as exact matches in the review or are "null" for implicit opinions.

Recognize all sentiment elements with their corresponding aspect terms, aspect categories, sentiment polarity, and opinion terms in the given input text (review). Provide your response in the format of a Python list of tuples: 'Sentiment elements: [("aspect term", "aspect category", "sentiment polarity", "opinion term"), ...]'. Note that ", ..." indicates that there might be more tuples in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.

"""

BASIC_PROMPT_ASTE = """According to the following sentiment elements definition:

- The "aspect term" refers to a specific feature, attribute, or aspect of a product or service on which a user can express an opinion. Explicit aspect terms appear explicitly as a substring of the given text.

- The "opinion term" refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service. Explicit opinion terms appear explicitly as a substring of the given text.

- The "sentiment polarity" refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Triplets with objective sentiment polarity should be ignored.

Please carefully follow the instructions. Ensure that aspect terms are recognized as exact matches in the review. Ensure that opinion terms are recognized as exact matches in the review. Ensure that sentiment polarities are from the available polarities.

Recognize all sentiment elements with their corresponding aspect terms, opinion terms, and sentiment polarity in the given input text (review). Provide your response in the format of a Python list of tuples: 'Sentiment elements: [("aspect term", "opinion term", "sentiment polarity"), ...]'. Note that ", ..." indicates that there might be more tuples in the list if applicable and must not occur in the answer. Ensure there is no additional text in the response.

"""

INSTRUCTIONS = {
    "acos/laptop16": BASIC_PROMPT_QUADRUPLET_TASKS.format(categories=LAPTOP_CATEGORIES),
    "acos/rest16": BASIC_PROMPT_QUADRUPLET_TASKS.format(categories=RESTAURANT_CATEGORIES),
    "asqp/rest15": BASIC_PROMPT_QUADRUPLET_TASKS.format(categories=RESTAURANT_CATEGORIES),
    "asqp/rest16": BASIC_PROMPT_QUADRUPLET_TASKS.format(categories=RESTAURANT_CATEGORIES),
    "aste/laptop14": BASIC_PROMPT_ASTE,
    "aste/rest14": BASIC_PROMPT_ASTE,
    "aste/rest15": BASIC_PROMPT_ASTE,
    "aste/rest16": BASIC_PROMPT_ASTE,
    "tasd/rest15": BASIC_PROMPT_TASD.format(categories=RESTAURANT_CATEGORIES),
    "tasd/rest16": BASIC_PROMPT_TASD.format(categories=RESTAURANT_CATEGORIES_TASD_REST16)
}
