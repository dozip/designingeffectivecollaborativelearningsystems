import os
import json
import logging
import shutil
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from pathlib import Path
from csv import writer
from datetime import datetime

from .agent import *
from .market import *
from .supply_chain import *
from helpers.helper_classes import *

# Def Logger
logger = logging.getLogger('example_logger')


class Reporting():
    """Class to structure the reporting and evaluation after the Simulation
    """

    def __init__(self, path, timestamp=None) -> None:
        self.path = Path(path)

        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        else:
            self.timestamp = timestamp

    def __init_reporting(self) -> None:
        """Create an Reporting Folder and CSV Overview
        """

        Path(self.path).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(data=[], columns=['time_stamp', 'Path'])
        self.path_csv = Path(self.path, "reporting_overview.csv")
        df.to_csv(self.path_csv, index=False, header=True)

    def __create_reporting_folder(self, run_id=None) -> None:
        """Create Folder for that simulation run
        """

        if run_id is None:
            self.path_report = Path(self.path, "simulation_" + str(self.timestamp))
            self.path_images = Path(self.path_report, "Images")
            self.path_data = Path(self.path_report, "Data")
            Path(self.path_report).mkdir(parents=True, exist_ok=True)
            Path(self.path_images).mkdir(parents=True, exist_ok=True)
            Path(self.path_data).mkdir(parents=True, exist_ok=True)
        else:
            self.path_report = Path(self.path, "simulation_" + str(self.timestamp))
            self.path_run = Path(self.path_report, "run_" + str(run_id))
            self.path_images = Path(self.path_run, "Images")
            self.path_data = Path(self.path_run, "Data")
            Path(self.path_report).mkdir(parents=True, exist_ok=True)
            Path(self.path_run).mkdir(parents=True, exist_ok=True)
            Path(self.path_images).mkdir(parents=True, exist_ok=True)
            Path(self.path_data).mkdir(parents=True, exist_ok=True)

        overview_info = [self.timestamp, self.path_report]
        self.report_overview_path = Path(self.path, "reporting_overview.csv")

        with open(self.report_overview_path, 'a') as file:
            writer_obj = writer(file)
            writer_obj.writerow(overview_info)
            file.close()

    def __create_market_reporting(self, market: Market) -> None:
        """Create the Report for the market

        Args:
            market (Market): _description_
        """

        demand = market.demand
        retailer_order = np.array(market.split_demand_history)
        n, m = retailer_order.shape
        columns = ["demand"]
        data = [demand]
        for j in range(m):
            columns.append("Demand_Retailer_" + str(j))

        for j in range(m):
            data.append(retailer_order[:, j])
        data = np.array(data).transpose()
        # data = [id_list, inventory, demand, order, supplier_order, supplier_order_new]

        df = pd.DataFrame(data=data, columns=columns)

        path = Path(self.path_data, "market.csv")
        df.to_csv(path, index=False, header=True)

        fig = plt.figure()
        plt.title("Market Demand and Retailer Demand over Time")
        plt.plot(df['demand'], label="market demand")
        plt.plot(df['Demand_Retailer_0'], label="market demand Retailer 0")
        plt.plot(df['Demand_Retailer_1'], label="market demand Retailer 1")
        plt.legend(loc=1)
        path = Path(self.path_images, "market_reporting.png")
        fig.savefig(path)
        plt.close()

    def __create_report_overview(self, cfg):
        """Create the report overview and parameters of the simulation run

        Args:
            cfg (_type_): _description_
        """

        # Serializing json
        json_object = json.dumps(cfg, indent=4)
        path = Path(self.path_report, "config.json")
        # Writing to sample.json
        with open(path, "w") as outfile:
            outfile.write(json_object)

    def __create_agent_reporting(self, agent_list: list[Agent]) -> None:
        """Create the report for all agents

        Args:
            agent_list (list[Agent]): _description_
        """

        agg_order_per_level = []  # aggregate the order per level per time step
        order_columns = []
        for k in range(len(agent_list)):  # go over each level of the SC
            agents = agent_list[k]  # select agents in level k
            agg_order = []  # aggr order
            order_list = []

            df_list = []
            # go over each agent i of SC level k
            for i, agent in enumerate(agents):
                # df with id, inventory, order, order per supplier, demand

                id = agent.id

                inventory = agent.inventory_history
                demand = agent.demand_sum_history
                demand_per_retailer = agent.demand_by_retailer_history
                forecast = agent.forecast_history
                forecast_by_retailer = agent.forecast_by_retailer_history 
                order = agent.order_history
                order_list.append(order)
                shipment = agent.received_shipment_history
                supplier_order = np.array(agent.order_per_supplier_history)
                n, m = np.array(supplier_order).shape

                columns = ["id", "inv", "demand", "forecast", "received_shipment", "order"]
                id_list = [id] * n
                data = [id_list, inventory, demand, forecast, shipment, order]
                for j in range(m):
                    columns.append("Order_Supplier_" + str(j))
                
                for j in range(agent.num_retailer):
                    columns.append("Forecast_Retailer_" + str(j))
                
                for j in range(agent.num_retailer):
                    columns.append("Demand_Retailer_" + str(j))

                for j in range(m):
                    data.append(supplier_order[:, j])
                
                for j in range(agent.num_retailer):
                    data.append(forecast_by_retailer[j])
                
                for j in range(agent.num_retailer):
                    data.append(demand_per_retailer[j])
                
                data = np.array(data).transpose()
                # data = [id_list, inventory, demand, order, supplier_order, supplier_order_new]

                df = pd.DataFrame(data=data, columns=columns)
                df_list.append(df)

                # Save Images
                fig = plt.figure()
                plt.title("Demand and Orders of Agent " +
                          str(agent.id) + " on Level " + str(k))
                plt.plot(df['demand'], label="demand")
                plt.plot(df['order'], label="order")
                plt.legend(loc=1)

                path = Path(self.path_images, "level_" + str(k) +
                            "_agent_" + str(agent.id) + "_demand_order.png")
                fig.savefig(path)
                plt.close()

                # fig = plt.figure()
                # plt.title("Demand and Split Demand of Agent " +str(agent.id) +" on Level " +str(k))
                # plt.plot(df['demand'], label = "demand")
                # for j in range(m):
                #     c_name = "Order_Supplier_"+str(j)
                #     plt.plot(df[c_name], label = "order supplier "+str(j))
                # plt.legend(loc = 1)
                # path = Path(self.path, "agent_"+str(agent.id)+"_demand_split_orders_on_level"+str(k)+".png")
                # fig.savefig(path)

            name = "agent_sc_level_" + str(k) + ".csv"
            self.path_report_retailer = Path(self.path_data, name)
            df_final = pd.concat(df_list)
            df_final.to_csv(self.path_report_retailer,
                            index=False, header=True)

            order_list = np.array(order_list)
            n, m = order_list.shape
            for e in range(m):
                agg_order.append(np.sum(order_list[:, e]))

            order_columns.append("agg_orders_level_" + str(k))
            agg_order_per_level.append(agg_order)
        agg_order_per_level = np.array(agg_order_per_level).transpose()
        df_order = pd.DataFrame(
            data=agg_order_per_level, columns=order_columns)
        path = Path(self.path_data, "agg_orders.csv")
        df_order.to_csv(path, index=False, header=True)

    def __create_BWE_reporting(self, agent_list: list) -> None:

        data = []
        columns = ['level', 'agent_id']
        for i, level in enumerate(agent_list):

            for j, agent in enumerate(level):

                BWE_measure = agent.variance_ratio

                data_ij = [[i], [j], BWE_measure]
                data_ij = [item for sublist in data_ij for item in sublist]
                data.append(data_ij)
        len_ = len(BWE_measure)
        for m in range(len_):
            header = "BWE_Measure_period_" + str(m)
            columns.append(header)
        df = pd.DataFrame(data=data, columns=columns)
        path = Path(self.path_data, "BWE_Measures.csv")
        df.to_csv(path, index=False)

    def __create_BWE_reporting_over_runs(self, agent_list: list, run_id: int) -> None:

        columns = ['run']
        data = [[run_id]]
        for i, level in enumerate(agent_list):

            for j, agent in enumerate(level):

                BWE_measure = agent.variance_ratio
                len_ = len(BWE_measure)
                data.append(BWE_measure)
                for m in range(len_):
                    header = "BWE_Measure_" + str(i) + "_" + str(j) + "_" + str(m)
                    columns.append(header)
        data = [item for sublist in data for item in sublist]
        df = pd.DataFrame(data=np.array(data)).transpose()
        df.columns = columns
        path = Path(self.path_report, "BWE_Measures_over_runs.csv")

        with open(path, 'a') as file:
            writer_obj = writer(file)
            writer_obj.writerow(data)
            file.close()
            
    def __create_training_reports(self, val_loss):
        df = pd.DataFrame(val_loss, columns = ['val_loss'])
        path = Path(self.path_data, "Validation_Loss.csv")
        df.to_csv(path, index=False)
        
        fig = plt.figure()
        plt.title("Validation_Loss")
        plt.plot(df['val_loss'], label="val_loss")
        plt.legend(loc=1)

        path = Path(self.path_images, "val_loss.png")
        fig.savefig(path)
        plt.close()
        

    def create_reporting(self, agent_list: list[Agent],
                         market: Market, supply_chain: Supply_Chain, cfg, val_loss):
        """Runs the entire reporting logic

        Args:
            agent_list (list[Agent]): _description_
            market (Market): _description_
            supply_chain (Supply_Chain): _description_
            cfg (_type_): _description_
        """

        try:
            logger.info("Create Reporting")

            if not os.path.isdir(self.path):
                self.__init_reporting()

            self.__create_reporting_folder()

            self.__create_market_reporting(market=market)
            
            self.__create_training_reports(val_loss)
        
            self.__create_agent_reporting(agent_list=agent_list)

            self.__create_BWE_reporting(agent_list=agent_list)

            self.__create_report_overview(cfg)
            

        except Exception as e:
            traceback.print_exc()
            df_delete = pd.read_csv(self.report_overview_path)
            df_delete = df_delete .drop(df_delete.index[-1])
            df_delete .to_csv(self.report_overview_path, index=False)
            shutil.rmtree(self.path_report)

    def create_reporting_multiple_runs(self, agent_list: list[Agent],
                                       market: Market, supply_chain: Supply_Chain, cfg, run_id, val_loss) -> None:

        if not os.path.isdir(self.path):
            self.__init_reporting()

        self.__create_reporting_folder(run_id=run_id)

        self.__create_report_overview(cfg)

        self.__create_market_reporting(market=market)

        self.__create_agent_reporting(agent_list=agent_list)
        
        self.__create_training_reports(val_loss)

        self.__create_BWE_reporting(agent_list=agent_list)

        self.__create_BWE_reporting_over_runs(agent_list=agent_list, run_id=run_id)
