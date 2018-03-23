# Neural-Machine-Translation
German to English, with attention!

#Getting started 
Download the data set http://www.manythings.org/anki/deu-eng.zip
Unzip it to the folder data/ and rename it to deu-eng.txt

Run python seq2seq.py and watch the neural translater learn!

#Some outputs
##Early in training, the translation is terrible
loss =  2.3753
sie ma ssen uns helfen .
you must help us .
you have to help you . <EOS>

##It starts to learn
loss = 1.5939
ich weia , wo sie lebt .
i know where she lives .
i know where she is . <EOS>

##Sometimes the translator gets angry
loss = 0.4927
pass auf , was du tust .
be careful what you do .
watch your mouth , do you work . <EOS>

##It starts learning the semantics
loss = 0.4510
ich bin ein wenig ma de .
i m a little bit tired .
i m a bit tired . <EOS>

##It's getting quite close, the translation is close to "It's not Tom's, yes?"
loss = 0.2933
es ist tom , nicht wahr ?
it s tom , isn t it ?
it s not tom s true ? <EOS>

##An exact match!
loss =  0.2407
ich lese ein sta ck .
i am reading a play .
i am reading a play . <EOS>

##The algorithm really learns the semantics. The correct sentence is much longer than the translation yet they carry the same meaning.
loss =  0.1976
ich habe zu viel zu tun .
i ve got too much to do .
i m too busy . <EOS>
