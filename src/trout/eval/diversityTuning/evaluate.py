import datasets
import torch

from trout.eval.diversityTuning.models.diversity_model import DiversityModel
from trout.metrics import DiversityScorer

ds = datasets.load_from_disk("data/diversityTuning/writingPrompt_cleaned")
val_dataset = ds["test"]
print(val_dataset)

# sample 1000
val_dataset = val_dataset.select(range(1000, 2000))

post_texts = val_dataset["post_text"]
post_titles = val_dataset["post_title"]
filtered_comment_texts = val_dataset["filtered_comment_texts"]
filtered_comment_scores = val_dataset["filtered_comment_scores"]
filtered_transformed_scores = val_dataset["filtered_transformed_scores"]

# for i in range(500):
#     # print(f"post_texts: {post_texts[i]}")
#     print(f"[POST TITLE] {post_titles[i]}\n")
#     # print(f"filtered_comment_texts: {filtered_comment_texts[i]}")
#     # print(f"filtered_comment_scores: {filtered_comment_scores[i]}")
#     # print(f"filtered_transformed_scores: {filtered_transformed_scores[i]}")
#     # print()

#     # print(len(filtered_comment_texts[i]))


texts1 = [
    "Why don’t cats play poker in the jungle? Too many cheetahs.",
    "I asked my cat if she wanted to hear a joke. She said, 'Paw-sibly.'",
    "What do you call a pile of kittens? A meow-tain.",
    "Why was the cat sitting on the computer? It wanted to keep an eye on the mouse.",
    "What do cats like to eat for breakfast? Mice Krispies.",
    "My cat's favorite button on the keyboard is paws.",
    "Why did the cat get kicked out of school? It was a real claw-en troublemaker.",
    "How do cats end a fight? They hiss and make up.",
    "Why did the cat bring a ladder? Because it wanted to reach the meow-sic notes.",
    "What’s a cat’s favorite movie? The Sound of Mewsic.",
]

texts2 = [
    "The stock market surged after better-than-expected earnings reports.",
    "What if dreams are just alternate timelines playing out?",
    "ERROR 404: The page you're looking for doesn't exist.",
    "He sprinted through the rain, clutching the last train ticket.",
    "Welcome to Cooking with Carla — today we're making spicy ramen!",
    "Congratulations! You've unlocked a secret achievement.",
    "Gravity pulls us all, but ambition makes us fly.",
    "Why do cats knock things off tables for no reason?",
    "Chapter 7: The Shadow in the Mirror",
    "Reminder: Your dentist appointment is tomorrow at 9:00 AM.",
]


semantic_diversity_model_name = "jinaai/jina-embeddings-v3"
style_diversity_model_name = "AnnaWegmann/Style-Embedding"
device = "cuda" if torch.cuda.is_available() else "cpu"

semantic_div_model = DiversityScorer(
    model_name=semantic_diversity_model_name, device=device
)
style_div_model = DiversityScorer(model_name=style_diversity_model_name, device=device)
avg_style_div1 = style_div_model.average_pairwise_diversity(texts1)
avg_sem_div1 = semantic_div_model.average_pairwise_diversity(texts1)
avg_style_div2 = style_div_model.average_pairwise_diversity(texts2)
avg_sem_div2 = semantic_div_model.average_pairwise_diversity(texts2)

print(f"Average Semantic Diversity Score 1: {avg_sem_div1}")
print(f"Average Style Diversity Score 1: {avg_style_div1}")
print(f"Average Semantic Diversity Score 2: {avg_sem_div2}")
print(f"Average Style Diversity Score 2: {avg_style_div2}")
