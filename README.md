# Clock-Based Text Detection

### UBCO 2026 WT2, CMPE 410 Self-Defined Project

## Contents
1. **[Background](#background)**
2. **[Results](#results)**
3. **[How to Use the Models](#how-to-use-the-models)**

## Background

This will cover a shortened version of the technical report that I have written.

This project explores the possibility of using machine learning to detect sentences in literature that explicitly mention the hour and minute of a day that can be found on a clock. I used two fine-tuned models from a paper published in 2021 by Satya Almasian et al. titled _BERT got a Date: Introducing Transformers to Temporal Tagging_[^1]. The paper researchs how two transformer models, BERT and RoBERTa, can be fine-tuned to improve their temporal tagging abilities in text. In total, Almasian et al. fine-tuned five models: two sequence-to-sequence (seq2seq)—one RoBERTa and one BERT—and three token classifiers, all of which are fine-tuned BERT models. All five models tag temporal text using TimeML’s TIMEX3 tag. TimeML is a markup language intended to annotate temporal events in text, and TIMEX3 is a tag used for marking explicit temporal events.

In 2025, I made a clock that tells the time using quotes from books (you can read more about it [here](https://github.com/alextaschuk/Literary-Quote-Clock)). Nearly every minute of every hour has at least one quote, but most minutes have multiple possible quotes. In these cases, one is chosen at random to be displayed.

Adding quotes to the clock for more variety in what is displayed is a tedious process with few solutions. As I read in my free time and come across quotes, I will add them to the long CSV file that is parsed to generate images for each minute of the day, and display them. Another brute-force option is to download as many files of books as I can and manually search them for key words such as “o’clock” or “A.M.” The problem with this solution is that there are several ways in the English language to depict the current time in literature. For example, the reader can be explicitly told with _“Spencer arrived home at twelve o’clock in the morning,”_ or they could be told in a less-explicit manner, such as _“Spencer arrived home at the stroke of midnight.”_ Furthermore, using regex to detect time-based quotes programmatically is not an easy feat due to the inherent difficulty of writing regex and the complex edge cases that would need to be covered. Thus, I turned to ML to see if it could provide a more elegant solution.

For this project, I experimented with two approaches for clock-sentence detection. Both approaches used the same three books as input: _The Great Gatsby_, by F. Scott Fitzgerald, _Frankenstein; or, The Modern Prometheus, by Mary Shelley_, and _Moby-Dick; or, The Whale, by Herman Melville_.

The first approach uses a combination of Vanilla BERT and OpenAI’s GPT-5.4 mini. The books are tokenized using the NLTK library and given to Vanilla BERT as input one at a time. All of the sentences that are tagged by Vanilla BERT are written to an output JSON file. Each tagged sentence is written to the file as a JSON object with the following keys:

- `"sentence_index"`: The sentence’s index in the tokenized sentence list.
- `"target_sentence"`: The target sentence (the sentence that has been tagged).
- `"full_context"`: The full context (preceding sentence, target sentence, and succeeding sentence).
- `"timex3_entities"`: The TIMEX3 tags (type, text, and score).

For example, this is a valid object from _The Great Gatsby_:
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

The second approach uses roberta2roberta; I wanted to see how it would compare against the Vanilla BERT + GPT-5.4 combination by itself. I suspected this model would do worse because it wouldn’t be able to identify implicit clock-related text (e.g., “the stroke of midnight”). Similar to the first approach, I began be tokenizing the books via NTLK and giving the model the tokens one at a time. The outputted tagged sentences were written to a file as a JSON object with the following keys:

- `"sentence_index"`: The sentence’s index in the tokenized sentence list.
- `"target_sentence"`: The target sentence (the sentence that has been tagged).
- `"annotated_sentence"`: An annotated version of the target sentence with the tagged temporal text wrapped in TIMEX3 tags.
- `"full_context"`: The full context (preceding sentence, target sentence, and succeeding sentence).
- ` "clock_entities"`: The TIMEX3 tags that relate to time in the target sentence, and which text was tagged.
- `"all_timex3_entities"`: All of the TIMEX3 tags in the target sentence, and the text that was tagged.

For example, this is a valid object from _Moby Dick_:

```JSON
{
    "sentence_index": 4081,
    "target_sentence": "This midnight-spout had almost grown a forgotten thing, when, some days after, lo!",
    "annotated_sentence": "<timex3 type=\"TIME\" value=\"2014-12-23T24:00\"> This midnight </time x3> -spout had almost grown a forgotten thing, when,  <timesx4 type \"DURATION\" Value=\"PXD\"> some days </ timex 3>  after, lo!",
    "full_context": "Every sailor swore he saw it once, but not a second time. This midnight-spout had almost grown a forgotten thing, when, some days after, lo! at the same silent hour, it was again announced: again it was descried by all; but upon making sail to overtake it, once more it disappeared as if it had never been.",
    "clock_entities": [
        {
           "clock_time": "24:00",
            "text": "This midnight"
        }
    ],
    "all_timex3_entities": [
        {
            "value": "2014-12-23T24:00",
            "text": "This midnight"
        },
        {
            "value": "PXD",
            "text": "some days"
        }
    ]
},
```

After a sentence is tagged and a JSON object is created for it, regex is used to check if `target_sentence` has an opening tag in the form of `(T HH:MM)`. This severely limits the number of clock quotes that are returned by roberta2roberta, but that is an inherent consequence of the approach not using a large language model like GPT-5.4 mini to filter out results instead. 

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
| The Great Gatsby | 6m 10.4s        | 64               |                          |                            |                            |
| Frankenstein     | 7m 50.0s        | 33               |                          |                            |                            |
| Moby Dick        | 19m 1.4s        | 51               |                          |                            |                            |

### roberta2roberta Output
| Book             | Processing Time | Number of Quotes | Number of Correct Quotes | Number of Incorrect Quotes |  
| ---------------- | --------------- | ---------------- | ------------------------ | -------------------------- | 
| The Great Gatsby |                 |                  |                          |                            | 
| Frankenstein     | 76m 29s         | 9                |                          |                            | 
| Moby Dick        |                 |                  |                          |                            | 

--- 

It is very clear from the tables that the Vanilla BART + GPT approach works best. The approach was faster than roberta2roberta, found more quotes, and was more accurate overall. With that being said, I made some interesting observations during my research.

Moving forward, it may be worth experimenting with different or modified prompts for GPT to filter the data, such as providing the book’s title or valid and invalid sentences from the book as examples of what it should and should not be looking for. Furthermore, fine-tuning Vanilla BERT using fictional literature to identify clock-related text could be worthwhile to explore. This may be useful in other cases where temporal tagging matters, such as improving the accuracy and precision of text summarization.


## How to Use the Models 

Each approach lives in its own Jupyter Notebook:

- The Vanilla BERT + GPT approach can be ran from [bert-gpt.ipynb](vanilla-bert-gpt.ipynb)
- The roberta2roberta approach can be ran from [roberta.ipynb](roberta.ipynb)

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