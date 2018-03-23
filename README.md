# Neural-Machine-Translation
German to English, with attention!

# Getting started 
Download the data set http://www.manythings.org/anki/deu-eng.zip
Unzip it to the folder data/ and rename it to eng-deu.txt

Run python seq2seq.py and watch the neural translator learn!

# Some outputs
## Early in training, the translation is terrible
loss =  2.3753

German = sie ma ssen uns helfen .

English = you must help us .

Translation = you have to help you . <EOS>

## It starts to learn
loss = 1.5939

German = ich weia , wo sie lebt .

English = i know where she lives .

Translation = i know where she is . <EOS>

## Sometimes the translator gets angry
loss = 0.4927

German = pass auf , was du tust .

English = be careful what you do .

Translation = watch your mouth , do you work . <EOS>

## It starts learning the semantics
loss = 0.4510

German = ich bin ein wenig ma de .

English = i m a little bit tired .

Translation = i m a bit tired . <EOS>

## It's getting quite close, the translation is close to "It's not Tom's, yes?"
loss = 0.2933

German = es ist tom , nicht wahr ?

English = it s tom , isn t it ?

Translation = it s not tom s true ? <EOS>

## An exact match!
loss =  0.2407

German = ich lese ein sta ck .

English = i am reading a play .

Translation =  i am reading a play . <EOS>

## The algorithm really learns the semantics. The correct sentence is much longer than the translation yet they carry the same meaning.
loss =  0.1976

German = ich habe zu viel zu tun .

English = i ve got too much to do .

Translation = i m too busy . <EOS>
