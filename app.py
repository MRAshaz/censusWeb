from flask import Flask, request, render_template, jsonify  # type: ignore
import pickle
import pandas as pd
import os

app = Flask(__name__)
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

port = int(os.environ.get("PORT", 5000))


@app.route("/")
def Home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    try:
        input_data = pd.DataFrame(
            [
                {
                    "age": int(data["age"]),
                    "workclass": data["workclass"],
                    "fnlwgt": int(data["fnlwgt"]),
                    "education": data["education"],
                    "education-num": int(data["education-num"]),
                    "marital-status": data["marital-status"],
                    "occupation": data["occupation"],
                    "relationship": data["relationship"],
                    "race": data["race"],
                    "sex": data["sex"],
                    "capital-gain": int(data["capital-gain"]),
                    "capital-loss": int(data["capital-loss"]),
                    "hours-per-week": int(data["hours-per-week"]),
                    "native-country": data["native-country"],
                }
            ]
        )

        prediction = model.predict(input_data)
        result = "Income > 50k" if str(prediction) in ["1", ">50k"] else "Income < 50k"
        return render_template("result.html", result=result)

    except KeyError as e:
        return jsonify({"error": f"Invalid category value: {e}"})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
