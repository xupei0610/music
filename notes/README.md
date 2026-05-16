All notes are extracted from Piano Fingering Dataset (PIG).
Please refer to the [official website of PIG](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/index.html) to learn more about the dataset.


Note files record the keyboard event.
The first line is used for description and can be anything or simply an empty line.

The following lines are organized as:

    NoteID  OnsetTime   OffsetTime  Pitch   OnsetVelocity   OffsetVelocity  Channel     Fingering

For example,

    0	0.00488284	1.00342	C5	69	80	0	1

defines a key pressing from 0.00488284s to 1.00342s on Key C5 with an onset velocity of 69 and an offset velocity of 80 on channel 0 using finger 1.

- The `NoteID` is just used for human users and will be ignored when the program parses it.
- The `Pitch` indicates the key being pressed. The supported pitch names include `A0`, `A#0`, `B0`, `C1`, `C#1`, `D1`, `D#1`, `E1`, `F1`, `F#1`, `G1`, `G#1`, `A1`, `A#1`, `B1`, `C2`, `C#2`, `D2`, `D#2`, `E2`, `F2`, `F#2`, `G2`, `G#2`, `A2`, `A#2`, `B2`, `C3`, `C#3`, `D3`, `D#3`, `E3`, `F3`, `F#3`, `G3`, `G#3`, `A3`, `A#3`, `B3`, `C4`, `C#4`, `D4`, `D#4`, `E4`, `F4`, `F#4`, `G4`, `G#4`, `A4`, `A#4`, `B4`, `C5`, `C#5`, `D5`, `D#5`, `E5`, `F5`, `F#5`, `G5`, `G#5`, `A5`, `A#5`, `B5`, `C6`, `C#6`, `D6`, `D#6`, `E6`, `F6`, `F#6`, `G6`, `G#6`, `A6`, `A#6`, `B6`, `C7`, `C#7`, `D7`, `D#7`, `E7`, `F7`, `F#7`, `G7`, `G#7`, `A7`, `A#7`, `B7`, `C8`, `Bb0`, `Db1`, `Eb1`, `Gb1`, `Ab1`, `Bb1`, `Db2`, `Eb2`, `Gb2`, `Ab2`, `Bb2`, `Db3`, `Eb3`, `Gb3`, `Ab3`, `Bb3`, `Db4`, `Eb4`, `Gb4`, `Ab4`, `Bb4`, `Db5`, `Eb5`, `Gb5`, `Ab5`, `Bb5`, `Db6`, `Eb6`, `Gb6`, `Ab6`, `Bb6`, `Db7`, `Eb7`, `Gb7`, `Ab7`, `Bb7`.
- `OnsetVelocity` and `OffsetVelocity` are defined as the volume definition in MIDI files, which are integers in the range of [0, 127]. Those values will be ignored, as the program currently does not support velocity control.
- `Fingering` defines the finger used for key pressing. It is an integer in the range from -5 to -1, representing fingers from the left little finger to the left thumb, and 1 to 5, representing fingers from the right thumb to the right little finger. Specially, we can leave it blank such that the program will use a nearest-finger heuristic during training to decide the fingering by itself. As a comparison, `017-fingering.txt` is from the PIG, which contains detailed fingering information, and `017-1_nofingering.txt` removes all the fingering information.
- `Channel` helps indicate the hand-level key event assignment when the `Fingering` field is blank. 0 indicates the right hand, and 1 indicates the left hand. A lazy setup could be using 0 for treble clef and 1 for bass clef.
- It is also allowed that `Channel` and `Fingering` are both blank, which means there is no predefined hand-level key assignment

