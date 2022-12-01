## RNN for predicting low glucose from gcm data

This is one of several models I use to predict events and guide treatment for my daughter's type 1 diabetes. I use this particular model to send text alerts to myself if a low event is predicted. 

I've been collecting gcm data from a [dexcom g6](https://www.dexcom.com/g6-cgm-system) since we received one in October.

Data is collected using [nightscout](https://github.com/nightscout/cgm-remote-monitor)

Data is stored in pg and combined with treatments and carb intake and queried using [nightscout_data_to_pg](https://github.com/patrickdmiller/nightscout_data_to_pg) which distributes the gcm readings to 1 sample every 5 minutes including carb and insulin intake.

A 3 layer (GRU32 x GRU64 x Dense 1) is trained using a window of 2 hours (24 samples) before a low or normal event within the next 2 readings (15 minute lookahead). Network is trained to classify for a low or normal event.

The model is served through a [flask/gunicorn/tf](https://github.com/patrickdmiller/docker_tensorflow_flask_gunicorn) stack api in my homelab. Every ~5 minutes when nightscout receives new data from dexcom, updates are pushed from nightscout > nodejs using [nightscout_socketio_client](https://github.com/patrickdmiller/nightscout_socketio_client). Nodejs sends new measures to the model api and takes action on the updated prediction (updates the dashboard and sends a text if an event is predicted)

~~see it in action [here](https://nsml.noice.us).~~ we have a pump now! upgrade to pulling data from glooko in progress

