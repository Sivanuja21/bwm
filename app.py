from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        num_factors = int(request.form['num_factors'])
        best_factor = int(request.form['best_factor'])
        worst_factor = int(request.form['worst_factor'])

        BO = {}
        OW = {}

        for i in range(1, num_factors + 1):
            if i != best_factor:
                BO[i] = int(request.form.get(f'bo_score_{i}', 0))
            if i != worst_factor:
                OW[i] = int(request.form.get(f'ow_score_{i}', 0))

        # Call the optimization function
        weights = optimize_bwm(num_factors, best_factor, worst_factor, BO, OW)
        
        # Render the results template
        return render_template('result.html', weights=weights)

    return render_template('index.html')

def optimize_bwm(num_factors, best_factor, worst_factor, BO, OW):
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

    prob = LpProblem("BWM_LP", LpMinimize)

    # Define the decision variables
    vars = LpVariable.dicts("W_D", (range(1, num_factors + 1)), 0, 1)
    KSI = LpVariable("KSI", 0)

    absb = LpVariable.dicts("ABS", (range(1, num_factors + 1)))  # Absolute Values for BO Constraints
    absw = LpVariable.dicts("ABSW", (range(1, num_factors + 1)))  # Absolute Values for OW Constraints

    # Sum of weights = 1
    prob += lpSum([vars[i] for i in range(1, num_factors + 1)]) == 1

    #### For BO
    for i in range(1, num_factors + 1):
        if i != best_factor:
            prob += absb[i] >= (vars[best_factor] - BO[i] * vars[i])
            prob += absb[i] >= -(vars[best_factor] - BO[i] * vars[i])
            prob += absb[i] <= KSI

    #### For OW
    for i in range(1, num_factors + 1):
        if i != worst_factor:
            prob += absw[i] >= (vars[i] - OW[i] * vars[worst_factor])
            prob += absw[i] >= -(vars[i] - OW[i] * vars[worst_factor])
            prob += absw[i] <= KSI

    # Defining the Objective Function
    prob += KSI

    # Solve the problem
    prob.solve()

    weights = {i: value(vars[i]) for i in range(1, num_factors + 1)}

    return weights

if __name__ == "__main__":
    app.run(debug=True)
