"""
The fun parts of BreezeML - in the spirit of a Japanese zen garden.

Like the guiding wind of Tsushima, BreezeML does not push you to the
destination; it shows you the way and lets you walk it.

    >>> import breezeml
    >>> breezeml.zen()        # the Zen of BreezeML - kaze no michi
    >>> breezeml.haiku()      # one machine learning haiku
    >>> breezeml.fortune()    # draw an omikuji for your model
"""
from __future__ import annotations

import random

__all__ = ["zen", "haiku", "fortune", "sensei"]


_SAKURA = r"""
                      .   *  .
               *   .      ~o~      .
          . ~o~     .  *    .   ~o~    *
        ,;;;,   .    ~o~  .    ,;;;,  .
       ;;;;;;;    ,;;;,      ;;;;;;;      *
    ~o~ ';;;'   ;;;;;;;   .   ';;;'  ~o~
   .     |      ';;;'  ~o~     |    .
         |    .   |       .    |         .
      ___|________|____________|___________
     /  kaze wa hana wo hakobu             \
    /   the breeze carries the petals       \
   '------------------------------------------'
"""

_ZEN_HAIKU = [
    ("test set sleeps in shade", "untouched until the last day", "honest numbers bloom"),
    ("wind does not push leaves", "it shows them where they should fall", "good defaults guide you"),
    ("a seed left unset", "the same garden never grows", "twice from the same soil"),
    ("the rare class hides deep", "like a fox in winter grass", "stratify, and see"),
    ("leakage is a stream", "silent under the temple", "loud splits keep it out"),
    ("four stones in the sand", "sklearn, pandas, numpy, joblib", "the garden needs none more"),
    ("perfect accuracy", "either the mountain is small", "or you brought the map"),
    ("naive baseline waits", "patient as the morning tide", "beat it, or bow out"),
    ("the median endures", "outliers howl like typhoons", "the middle stays calm"),
    ("export the model", "a bird kept in an open cage", "stays because it wants"),
    ("drift arrives at dusk", "wearing last season's kimono", "watch the distributions"),
    ("tune with patience, friend", "a thousand random gardens grow", "one holds the right depth"),
    ("ship the model card", "a sword is judged by its edge", "a model by its caveats"),
    ("deep nets on ten rows", "a katana to slice tofu", "put the blade away"),
    ("the gradient descends", "the mountain does not notice", "did it find the valley?"),
]

# Omikuji: traditional Japanese fortune slips, drawn at shrines.
_OMIKUJI = [
    ("Dai-kichi (Great Blessing)", [
        "Your splits are stratified, your seeds are set. Ship with a calm heart.",
        "The leaderboard smiles upon you. The naive baseline bows in defeat.",
        "Today your pipeline is leak-free. Honor it with a model card.",
    ]),
    ("Kichi (Blessing)", [
        "A good model comes to those who cross-validate.",
        "Your features are clean. Your holdout is honest. Proceed.",
        "The guiding wind favors gradient boosting today.",
    ]),
    ("Chuu-kichi (Middle Blessing)", [
        "The model is adequate. Like tea that is merely warm, it will serve.",
        "R2 of 0.7: not a masterpiece, not a disgrace. The middle path.",
        "Tune one more hour, no more. Then accept what the garden gives.",
    ]),
    ("Shou-kichi (Small Blessing)", [
        "A small improvement waits in feature engineering. Dig quietly.",
        "Your accuracy hides a shy macro F1. Look closer at the rare class.",
        "The forecast beats naive by a whisker. A whisker is still a win.",
    ]),
    ("Kyou (Misfortune)", [
        "Beware: a test set has been touched twice. The numbers now flatter.",
        "Drift stirs in production like wind before a storm. Check /drift.",
        "Perfect training accuracy, weeping validation. The oldest tragedy.",
        "Somewhere a target column leaked into the features. Find it before it finds you.",
    ]),
]


def _speak(text: str) -> None:
    """Print with a plain-ASCII fallback for terminals stuck in the past."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "ignore").decode())


def zen() -> str:
    """The Zen of BreezeML: kaze no michi, the way of the wind."""
    lines = [_SAKURA, "        The Zen of BreezeML", "        ~ kaze no michi : the way of the wind ~", ""]
    for first, second, third in _ZEN_HAIKU:
        lines.append(f"    {first},")
        lines.append(f"      {second} -")
        lines.append(f"        {third}.")
        lines.append("")
    lines.append("    Like the guiding wind: it does not push you,")
    lines.append("    it shows the way, and lets you walk it.")
    text = "\n".join(lines)
    _speak(text)
    return text


def haiku(seed: int | None = None) -> str:
    """One machine learning haiku, carried in on the breeze."""
    rng = random.Random(seed)
    first, second, third = rng.choice(_ZEN_HAIKU)
    text = f"{first},\n  {second} -\n    {third}."
    _speak(text)
    return text


_DOJO = r"""
              ____________________
             /                    \
            /   B R E E Z E M L    \
           /        D O J O         \
          '--------------------------'
              |    |        |    |
              |    |  ~o~   |    |
          ____|____|________|____|____
         (____________________________)

        Sensei Akash Anipakalu Giridhar
        ~ founder of the BreezeML dojo ~
"""

_TEACHINGS = [
    "Four dependencies. Zero excuses.",
    "A student asked: 'Sensei, my accuracy is perfect.' Sensei replied: 'Then your test set is not.'",
    "First learn fit. Then learn predict. Then learn why the split matters. This is the way.",
    "Sensei does not fear the strong model. Sensei fears the unexamined one.",
    "When you can export the pipeline and walk away, only then have you truly stayed.",
    "The novice tunes hyperparameters. The master tunes the question.",
    "A student asked: 'Which model is best?' Sensei pointed at compare() and said nothing.",
    "Wind cannot be owned, only followed. So too a good default.",
    "Sensei shipped on Friday once. Once.",
    "Before deploying, write the model card. Before the model card, understand the caveats. Before the caveats, humility.",
]


def sensei(seed: int | None = None) -> str:
    """Seek the founder of the dojo. Receive one teaching.

    Parameters
    ----------
    seed : int, optional
        The teaching you receive is the teaching you were meant to receive.
        Unless you set a seed. Then it is reproducible, as sensei prefers.
    """
    rng = random.Random(seed)
    teaching = rng.choice(_TEACHINGS)
    text = f"{_DOJO}\n        Sensei says:\n        \"{teaching}\"\n"
    _speak(text)
    return text


def fortune(seed: int | None = None) -> str:
    """Draw an omikuji (shrine fortune slip) for your machine learning day.

    Parameters
    ----------
    seed : int, optional
        Set for a reproducible fortune - the only honest kind.
    """
    rng = random.Random(seed)
    rank, readings = rng.choice(_OMIKUJI)
    reading = rng.choice(readings)
    text = f"~ omikuji ~\n[{rank}]\n{reading}"
    _speak(text)
    return text
