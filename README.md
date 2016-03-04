# beaver

## About ##
Beaver is a proof of concept final project (written in C and C++) conceived of and completed by Tellis Bethel as a requirement for the Acadia University Computer Science Department course COMP 4983, under the supervision of Dr. Darcy Benoit.

Simply put, Beaver is a separated single source electronic drum set, unlike a normal electronic drum kit/set wherein sensors found in each component pad send signals to trigger a pad's associated audio sample. Although Beaver's usage is analogous to electronic drum kit usage in terms of object striking and audio sample playback, a number of key differences exist.

Firstly, whereas an electronic drum kit consists of several component pads (made of rubber), Beaver allows users to define his/her own drum set with percussive objects that selected from his/her vicinity. Secondly, though an electronic drum set requires a sensor in each component pad to trigger the playback device, Beaver has one sensor—a microphone—which gathers audio data. This microphone processes the data to determine which user-defined component has been struck, triggering the playback of the associated audio sample. Therefore, unlike an electronic drum kit/set, Beaver's sensor is not bound to the strike location.  

The implementation of Beaver intended to determine whether one could successfully classify user-defined percussive objects using advanced signal processing and standard machine learning methods. These signal processing methods needed to enable the application to trigger corresponding audio samples in real time. Real time, for the purposes of this project, is defined as a round-trip latency of no more than 15 milliseconds (ms).

## Usage ##
See 'BeaverWalkthrough.pdf'
