from Loading_and_Prossening import get_vtb_data_moex, prepare_features
from Neuron_network import load_data, sample_splitting, graph, forecast_future

def run():
    raw_df = get_vtb_data_moex()
    prepare_features(raw_df)
    df = load_data()
    predictions, act_prices, model, sc_y, x_test_data = sample_splitting(df)
    future_rubles = forecast_future(model, x_test_data, sc_y, 22)
    graph(act_prices, predictions, future_rubles)
if __name__ == "__main__":
    run()