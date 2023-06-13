import eel

modelConfig = {}

eel.init("web", allowed_extensions=['.js', '.html', '.png', '.txt'])


@eel.expose
def save_model_config(x):
    for key in x:
        try:
            modelConfig[key] = int(x[key])
        except:
            try:
                modelConfig[key] = float(x[key])
            except:
                modelConfig[key] = x[key]
    print(modelConfig)


eel.setStrings("hey")
eel.start("index.html")

