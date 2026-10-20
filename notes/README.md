# Note files

This folder holds the scores that the piano tasks play (`main.py --note <file>`).
All scores come from the [PIG Piano Fingering Dataset](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/index.html).
Refer to the PIG website for the dataset and its license.

## Files

Each piece has two versions:

| Suffix | Content | Training task |
|---|---|---|
| `_fingering.txt` | The PIG fingering: one finger for each note | The policy must press each key with the given finger. |
| `_hand.txt` | The same notes with no fingering. `Channel` gives the hand of each note. | The policy selects the finger in the given hand. |

The pretrained policies in `pretrained/joint_fingering/` and `pretrained/joint_hand/` use the file with the same name.
For example, `pretrained/joint_hand/052-1_hand.ckpt` plays `notes/052-1_hand.txt`.

| PIG ID | Piece | Length |
|---|---|---|
| 017-1 | Mozart, Piano Sonata K. 545 in C major, 1st movement | 23.5 s |
| 046-1 | Bach, French Suite No. 3, Minuet | 15.7 s |
| 048-1 | Bach, English Suite No. 2, Prelude | 27.3 s |
| 052-1 | Mozart, Piano Sonata K. 280 in F major, 2nd movement | 15.9 s |
| 061-2 | Beethoven, Für Elise | 19.9 s |
| 070-1 | Beethoven, Piano Sonata No. 23, 2nd movement | 15.6 s |
| 075-1 | Chopin, Waltz Op. 64 No. 2 | 19.2 s |
| 087-1 | Debussy, Suite bergamasque, Passepied | 25.8 s |
| 092-1 | Grieg, Lyric Pieces Op. 38 No. 1 | 16.0 s |
| 102-1 | Mussorgsky, Pictures at an Exhibition, Bydło | 20.0 s |
| 117-1 | Brahms, Rhapsody Op. 79 No. 2 | 24.0 s |
| 119-1 | Joplin, Maple Leaf Rag | 20.0 s |
| 123-1 | Beethoven, Piano Sonata No. 3, 2nd movement | 16.0 s |
| 140-1 | Scarlatti, Sonata in A major K. 208 | 14.7 s |
| 145-1 | Scriabin, Piano Sonata No. 5 | 31.4 s |

## Format

The files use the PIG text format, with one note on each line:

    NoteID  OnsetTime  OffsetTime  Pitch  OnsetVelocity  OffsetVelocity  Channel  Fingering

The fields are separated by tabs or spaces. For example:

    0	0.00488284	1.00342	C5	69	80	0	1

This line presses C5 from 0.00488 s to 1.00342 s with the right thumb.
The parser skips blank lines and lines that start with `//`, for example the PIG header `//Version: PianoFingering_v170101`.
Do not put any other text in the file, because the parser reads every other line as a note.

| Field | Meaning |
|---|---|
| `NoteID` | A label for people. The parser does not use it. |
| `OnsetTime`, `OffsetTime` | The start and end of the key press, in seconds. |
| `Pitch` | A note name from `A0` to `C8`. Sharps use `#`, for example `C#4`. The flats `Db`, `Eb`, `Gb`, `Ab` and `Bb` are also accepted. A MIDI note number from 21 to 108 is also accepted. |
| `OnsetVelocity`, `OffsetVelocity` | MIDI velocities from 0 to 127. The parser does not use them, because the tasks have no velocity control. |
| `Channel` | The hand: `0` is the right hand and `1` is the left hand. Any other integer, for example `-1`, means "no hand given". |
| `Fingering` | The finger: `1` to `5` are the right thumb to the right little finger. `-1` to `-5` are the left thumb to the left little finger. |

A line can have 6, 7 or 8 fields:

| Fields | Meaning | During training |
|---|---|---|
| 8 | Hand and finger are given. | The reward uses the given finger. |
| 7 (no `Fingering`) | Only the hand is given. This is the format of the `_hand.txt` files. | The reward uses the nearest finger of the given hand. |
| 6 (no `Channel`, no `Fingering`) | Nothing is given. | The reward uses the nearest finger of either hand. |

If `Fingering` is not a value from the list above, the parser uses only `Channel` for that note.
For example, PIG writes a finger change on a held note as `3_1`.

## How the parser converts the times

- It moves the first onset of the file to time 0.
- It rounds each onset and offset to the nearest control frame. The tasks run at 60 frames per second.
- It skips a note that is shorter than 3 frames (50 ms) and prints a message.
- If a note follows another note on the same key with no rest, the parser releases the key for one frame between the two notes. This makes the hand lift the key and press it again.
