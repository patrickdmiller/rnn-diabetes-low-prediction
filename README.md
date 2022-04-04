## RNN for predicting low glucose from gcm data

I've been collecting gcm data from a [dexcom g6](https://www.dexcom.com/g6-cgm-system) since my daughter received one in October.

Data is collected using [nightscout](https://github.com/nightscout/cgm-remote-monitor)

Data is stored in pg and combined with treatments and carb intake and queried using [nightscout_data_to_pg](https://github.com/patrickdmiller/nightscout_data_to_pg) which distributes the gcm readings to 1 sample every 5 minutes including carb and insulin intake.

A 3 layer (GRU32 x GRU64 x Dense 1) is trained using a window of 2 hours (24 samples) before a low or normal event within the next 2 readings (10 minute lookahead). Network is trained to classify for low or normal event.

model is loaded into node.js using tensorflow-js. when a new reading comes in, status is updated and notifications sent. 

see it in action [here](https://nsml.noice.us)
