# Clock-Based Text Detection

### UBCO 2026 WT2, CMPE 410 Self-Defined Project

## Contents
1. **[Background](#background)**
2. **[Results](#results)**
3. **[How to Use the Models](#how-to-use-the-models)**

## Background

This will cover a shortened version of the technical report that I have written. For the full report, see [report.pdf](/report.pdf)

You can find the models' output for each book in the respective [frankenstein](/frankenstein/), [moby_dick](/moby_dick/), and [the_great_gatsby](/the_great_gatsby/) folders.

This project explores the possibility of using machine learning to detect sentences in literature that explicitly mention the hour and minute of a day that can be found on a clock. I used two fine-tuned models from a paper published in 2021 by Satya Almasian et al. titled _BERT got a Date: Introducing Transformers to Temporal Tagging_.[^1] The paper researchs how two transformer models, BERT and RoBERTa, can be fine-tuned to improve their temporal tagging abilities in text. In total, Almasian et al. fine-tuned five models: two sequence-to-sequence (seq2seq)—one RoBERTa and one BERT—and three token classifiers, all of which are fine-tuned BERT models. All five models tag temporal text using TimeML’s TIMEX3 tag. TimeML is a markup language intended to annotate temporal events in text, and TIMEX3 is a tag used for marking explicit temporal events.

In 2025, I made a clock that tells the time using quotes from books (you can read more about it [here](https://github.com/alextaschuk/Literary-Quote-Clock)). Nearly every minute of every hour has at least one quote, but most minutes have multiple possible quotes. In these cases, one is chosen at random to be displayed.

Adding quotes to the clock for more variety in what is displayed is a tedious process with few solutions. As I read in my free time and come across quotes, I will add them to the long CSV file that is parsed to generate images for each minute of the day, and display them. Another brute-force option is to download as many files of books as I can and manually search them for key words such as “o’clock” or “A.M.” The problem with this solution is that there are several ways in the English language to depict the current time in literature. For example, the reader can be explicitly told with _“Spencer arrived home at twelve o’clock in the morning,”_ or they could be told in a less-explicit manner, such as _“Spencer arrived home at the stroke of midnight.”_ Furthermore, using regex to detect time-based quotes programmatically is not an easy feat due to the inherent difficulty of writing regex and the complex edge cases that would need to be covered. Thus, I turned to ML to see if it could provide a more elegant solution.

For this project, I experimented with two approaches for clock-sentence detection. Both approaches used the same three books as input: _The Great Gatsby_, by F. Scott Fitzgerald, _Frankenstein; or, The Modern Prometheus_, by Mary Shelley_, and _Moby-Dick; or, The Whale, by Herman Melville_.[^2]

The first approach uses a combination of Vanilla BERT and OpenAI’s GPT-5.4 mini. The books are tokenized using the NLTK library and given to Vanilla BERT as input one at a time. All of the sentences that are tagged by Vanilla BERT are written to an output JSON file. Each tagged sentence is written to the file as a JSON object with the following keys:

- `"sentence_index"`: The sentence’s index in the tokenized sentence list.
- `"target_sentence"`: The target sentence (the sentence that has been tagged).
- `"full_context"`: The full context (preceding sentence, target sentence, and succeeding sentence).
- `"timex3_entities"`: The TIMEX3 tags (type, text, and score).

For example, this is a tagged object for text from  _The Great Gatsby_:
```JSON
{
    "sentence_index": 99,
    "target_sentence": "Tomorrow!” Then she added irrelevantly: “You ought to see the baby.” “I’d like to.” “She’s asleep.",
    "full_context": "Let’s go back, Tom. Tomorrow!” Then she added irrelevantly: “You ought to see the baby.” “I’d like to.” “She’s asleep. She’s three years old.",
    "timex3_entities": [
        {
            "type": "DATE",
            "text": "tomorrow",
            "score": 0.9982
        }
    ]
},
```

Next, GPT-5.4 mini is used to filter out the non-clock-related sentences. The model was prompted to determine whether a specific clock time is present and the minimal excerpt from the full context needed to make the sentence meaningful, along with the $n^{th}$ JSON object from Vanilla BERT’s output file. JSON objects were input one at a time. The filtered results were written to an output JSON file in the same format as Vanilla BERT’s output file.

For example, this is an object for text from  _The Great Gatsby_:
```JSON
{
    "sentence_index": 2048,
    "target_sentence": "“I’ll call you about noon.”",
    "full_context": "“I’ll call you up,” I said finally. “Do, old sport.”  “I’ll call you about noon.”  We walked slowly down the steps. “I suppose Daisy’ll call too.” He looked at me anxiously, as if he hoped I’d corroborate this.",
    "timex3_entities": [
        {
            "type": "TIME",
            "text": "noon",
            "score": 0.9673
        }
    ]
},
```

The second approach uses roberta2roberta; I wanted to see how it would compare against the Vanilla BERT + GPT-5.4 combination by itself. Even though roberta2roberta was supposed to be a strong model, I suspected that it would perform worse because of the strict filtering that would be required without an LLM. Similar to the first approach, I began be tokenizing the books via NTLK and giving the model the tokens one at a time. The outputted tagged sentences were written to a file as a JSON object with the following keys:

- `"sentence_index"`: The sentence’s index in the tokenized sentence list.
- `"target_sentence"`: The target sentence (the sentence that has been tagged).
- `"annotated_sentence"`: An annotated version of the target sentence with the tagged temporal text wrapped in TIMEX3 tags.
- `"full_context"`: The full context (preceding sentence, target sentence, and succeeding sentence).
- ` "clock_entities"`: The TIMEX3 tags that relate to time in the target sentence, and which text was tagged.
- `"all_timex3_entities"`: All of the TIMEX3 tags in the target sentence, and the text that was tagged.

After a sentence is tagged and a JSON object is created for it, regex is used to check if `target_sentence` has an opening tag in the form of `(T HH:MM)`. This severely limits the number of clock quotes that are returned by roberta2roberta, but that is an inherent consequence of the approach not using a large language model like GPT-5.4 mini to filter out results instead. 

For example, this is a valid object from _Moby Dick_:

```JSON
{
    "sentence_index": 5995,
    "target_sentence": "By midnight the works were in full operation.",
    "annotated_sentence": "By <timex3 type=\"TIME\" value=\"1998-12-08T24:00\"> midnight </timeXX </timesx 3>  the works were in full operation.  . <TIME\"[ value \"1998\"> noon </x2> .",
    "full_context": "It smells like the left wing of the day of judgment; it is an argument for the pit. By midnight the works were in full operation. We were clear from the carcase; sail had been made; the wind was freshening; the wild ocean darkness was intense.",
    "clock_entities": [
        {
            "clock_time": "24:00",
            "text": "midnight"
        }
    ],
    "all_timex3_entities": [
        {
            "value": "1998-12-08T24:00",
            "text": "midnight"
        }
    ]
},
```

## Results


### Vanilla BERT Output
| Book             | Processing Time | Number of Quotes Tagged |
| ---------------- | --------------- | ----------------------- |
| The Great Gatsby | 33.1s           | 514                     |
| Frankenstein     | 43.6s           | 634                     |
| Moby Dick        | 1m 47.9s        | 1,479                   |

### GPT-5.4 mini Post-Filtering Output
| Book             | Processing Time | Number of Quotes | Number of Correct Quotes | Number of Incorrect Quotes | Number of Duplicate Quotes |
| ---------------- | --------------- | ---------------- | ------------------------ | -------------------------- | -------------------------- |
| The Great Gatsby | 6m 10.4s        | 64               | 62                       | 2                          | 19                         |
| Frankenstein     | 7m 50.0s        | 33               | 32                       | 1                          | 10                         |
| Moby Dick        | 19m 1.4s        | 51               | 47                       | 4                          | 5                          |

### roberta2roberta Output
| Book             | Processing Time | Number of Quotes | Number of Correct Quotes | Number of Incorrect Quotes | 
| ---------------- | --------------- | ---------------- | ------------------------ | -------------------------- | 
| The Great Gatsby | 55m 15.4s       | 22               | 16                       | 6                          | 
| Frankenstein     | 76m 29s         | 9                | 6                        | 3                          | 
| Moby Dick        | 209m 24.0s      | 34               | 27                       | 7                          | 

--- 

It is very clear from the tables that the Vanilla BART + GPT approach works best. The approach was faster than roberta2roberta, found more quotes, and was more accurate overall. That said, I noticed that roberta2roberta tagged a fair number of sentences that VB overlooked or GPT filtered out. 

Moving forward, it may be worth experimenting with different or modified prompts for GPT to filter the data, such as providing the book’s title or valid and invalid sentences from the book as examples of what it should and should not be looking for. Furthermore, fine-tuning Vanilla BERT using fictional literature to identify clock-related text could be worthwhile to explore. This may be useful in other cases where temporal tagging matters, such as improving the accuracy and precision of text summarization.


## How to Use the Models 

Each approach lives in its own Jupyter Notebook:

- The Vanilla BERT + GPT approach can be ran from [bert-gpt.ipynb](/vanilla-bert-gpt.ipynb)
- The roberta2roberta approach can be ran from [roberta.ipynb](/roberta.ipynb)

The first two steps for both approaches is the same:
1. Clone the repository locally.
2. In the `Imports & Config` cell, modify `BOOK_FILE` to store the filepath to the book you want to get clock quotes from.

### Running Vanilla BERT + GPT

3. Go to OpenAI's [API Platform](https://platform.openai.com/) and make a new project.
4. Create a new secret key for the project.
5. Create a `.env` file and store the secret key in a variable called `OPENAI_API_KEY`.
6. Run the notebook. Two output files will be created:
    - `bert_candidates.json`: Contains all of the sentences that Vanilla BERT tagged.
    - `gpt_filtered.json`: Contains the objects from `bert_candidates.json` that GPT determined to contain clock-related text.

### Running roberta2roberta

3. Run the notebook. One output file will be created:
    - `roberta_tagged.json`


[^1]: The repository for Almasian et al's paper can be found [here](https://github.com/satya77/Transformer_Temporal_Tagger).

[^2]: You can find the downloads for _The Great Gatsby_ [here](https://www.gutenberg.org/ebooks/64317), _Frankenstein_ [here](https://www.gutenberg.org/ebooks/84), and _Moby Dick_ [here](https://www.gutenberg.org/ebooks/2701).