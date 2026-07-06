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

__all__ = ["zen", "haiku", "fortune"]


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
