from dataclasses import dataclass
from typing import Optional

import pandas as pd
import datetime
import random

@dataclass
class Order:  #our own placed orders
    order_id: int
    size: float
    price: float
    timestamp: datetime 
    side: str

@dataclass
class AnonTrade:  #trades from market data
    timestamp: datetime
    size: float
    price: str
    side: str 
    

@dataclass
class OwnTrade:  #successful trades
    timestamp: datetime
    trade_id: int 
    order_id: int
    size: float
    price: float
    side: str


@dataclass
class OrderbookSnapshotUpdate:  #current orderbook's shapshot
    timestamp: datetime
    asks: list[tuple[float, float]]  
    bids: list[tuple[float, float]]


@dataclass
class MdUpdate:  #data of a tick
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trades: Optional[list[AnonTrade]] = None


class Strategy:
    def __init__(self, max_position: float, t_0: int, maker_fee: int, pnl_data) -> None:
        #we want to control maximum size of inventory that we have after each execution
        self.max_position = max_position 
        #if order is waiting for placing more that t_0, than we cancel this order
        self.t_0 = t_0
        self.total_size = 0.0 #total size of our inventory
        self.pnl = 0 #our profits and losses
        self.size = 0.001 #size of each order
        self.maker_fee = maker_fee #fee that exchange takes
        self.pnl_data = pnl_data #dependence between pnl and orderbook.timestamp
        self.temp_bids = [] #successfully executed bids
        self.temp_asks = [] #successfully executed asks

    def run(self, sim: "Sim"):
        while True:
            try:
                orderbook, own_trades, orders_dict_latency = sim.tick(self.maker_fee)
                
                #if this snapshot is the last one:
                if orderbook.timestamp == 1656028781362646546:

                    #while each of temp_bids and temp_asks is not none,
                    #we may calculate pnl by comparing these two orders 
                    while len(self.temp_bids)*len(self.temp_asks) > 0:
                        
                        self.pnl += self.temp_asks[0] - self.temp_bids[0]
                        self.temp_asks.pop(0)
                        self.temp_bids.pop(0)

                    #when one of these lists is none,
                    #we execute other side aggressively 
                    #(sell or buy by orderbook price)
                    if len(self.temp_asks) == 0:
                        while len(self.temp_bids)>0:
                            self.pnl += ((orderbook.asks[0]) - self.temp_bids[0])*self.size
                            self.temp_bids.pop(0)
                    else:
                        while len(self.temp_asks)>0:
                            self.pnl += ((-orderbook.bids[0]) + self.temp_asks[0])*self.size

                    self.total_size = (len(self.temp_bids) - len(self.temp_asks))*self.size

                #we randomly choose the side (bid or ask)    
                side_ = random.randint(0,1)

                #split own_trades on bids and asks
                for order_id, order in own_trades.copy().items():

                    if order.side == 'BID':
                        self.temp_bids.append(order.price)

                    elif order.side == 'ASK':
                        self.temp_asks.append(order.price)

                    own_trades.pop(order_id)

                while len(self.temp_bids)*len(self.temp_asks)>0:

                    #we can calculate pnl only if we have bid AND ask executed orders
                    self.pnl += (self.temp_asks[0] - self.temp_bids[0])*self.size
                    self.temp_asks.pop(0)
                    self.temp_bids.pop(0)
                
                #calculating the total size of inventory
                self.total_size = (len(self.temp_bids) - len(self.temp_asks))*self.size
                
                #max_position control 
                #if total size MORE that max_position, then we can't buy more assets
                if (self.total_size <= self.max_position) and (side_ == 0): 
                    side = 'BID'
                    price = orderbook.bids[0]
                    sim.place_order(side, self.size, price, orderbook.timestamp)
                #if total size LESS than max_position*(-1), then we can't sell more assets
                elif (self.total_size >= self.max_position * (-1)) and (side_ == 1): 
                    side = 'ASK'
                    price = orderbook.asks[0]
                    sim.place_order(side, self.size, price, orderbook.timestamp)

                #cancel orders, that haven't exposed for t_0
                for order_id, order in orders_dict_latency.copy().items():
                    if (orderbook.timestamp - order.timestamp) >= (self.t_0)*1000000:
                        sim.cancel_order(order_id)
                    else:
                        break 
                self.pnl_data.append([self.pnl, orderbook.timestamp, self.total_size, orderbook.bids[0]])
            except StopIteration:
                return self.pnl_data
                break
        


def load_md_from_file(path: str) -> list[MdUpdate]:
    #load data
    btc_lobs =  pd.read_csv(f'{path}/lobs.csv')
    btc_trades = pd.read_csv(f'{path}/trades.csv')

    #save only best bid/ask prices and vols
    btc_lobs = btc_lobs.set_index('receive_ts', drop=False)
    btc_lobs = btc_lobs.iloc[:, :6] 
    btc_lobs.columns = [i.replace('btcusdt:Binance:LinearPerpetual_', '') for i in btc_lobs.columns] 

    #merge data (because we have to different columns: 'receive_ts' and 'exchange_ts')
    md = btc_trades.groupby(by='exchange_ts').agg({'receive_ts': 'last', 'price': 'max', 'aggro_side': ['last', 'count']})
    md.columns = ['_'.join(i) for i in md]
    df = pd.merge_asof(md, btc_lobs.iloc[:, 2:6], left_on='receive_ts_last', right_index=True)

    #make a slice of your data (optional)
    #df = df.iloc[:100000, :]
    
    #lists for shapshots of orderbook and trades
    orderbooks = []
    trades = []

    asks = df[['ask_price_0', 'ask_vol_0']].values
    bids = df[['bid_price_0', 'bid_vol_0']].values
    receive_ts_ = df['receive_ts_last'].values

    #append current orderbook/trade into orderbooks/trades
    for i in range(df.shape[0]):
        orderbook = OrderbookSnapshotUpdate(receive_ts_[i], asks[i], bids[i])
        orderbooks.append(orderbook)
        trade = AnonTrade(receive_ts_[i], 0.001, df['price_max'].values[i], df['aggro_side_last'].values[i])
        trades.append(trade)
        
    return orderbooks, trades


class Sim:
    def __init__(self, execution_latency: int, md_latency: int) -> None:
        self.orderbooks, self.trades = load_md_from_file("..md/btcusdt_Binance_LinearPerpetual")
        self.orderbook = iter(self.orderbooks)
        self.trade = iter(self.trades)
        #execution_latency - delay due to the fact that orders are not placed on the market instantly  
        self.execution_latency = execution_latency
        #md_latency - delay due to the fact that we don't receive information about successful trades instantly
        self.md_latency = md_latency
        self.orders_dict = {} #orders in queue for placing in orderbook
        self.orders_dict_latency = {} #orders that we are placing on exchange 
                                      #(each order 've waited execution_latency time)
        self.own_trades = {} #successfully executed trades
        self.trade_id = 1 #ids for successfully executed trades
        self.order_id = 1 #ids for trades that are on exchange now

    #this function imitates real behaviour of orderbook on exchange
    #it returns what 've happened for one tick of orderbook
    def tick(self, maker_fee) -> MdUpdate:
        trade = next(self.trade)
        orderbook = next(self.orderbook)
        #if order 've waited for execution_latency time, it must be exposed on exchange
        for order_id, order in self.orders_dict.copy().items():
            #.timestamp is in nanosecs, but latency is in ms, so we multiply latency by 10^6 
            if (orderbook.timestamp - order.timestamp) >= (self.execution_latency)*1000000:
                self.prepare_orders(order_id, order)
                #delete our order from orders_dict (because now this order is on exchange)
                self.orders_dict.pop(order_id) 

        own_trades = self.execute_orders(trade, orderbook, maker_fee)  

        return orderbook, own_trades, self.orders_dict_latency

    #this function exposes current order on exchange
    #(this order is on exchange now)
    def prepare_orders(self, order_id, order):
        self.orders_dict_latency[order_id] = order

        return self.orders_dict_latency
    
    #this function executes our orders 
    #(on specific conditions)
    def execute_orders(self, trade, orderbook, maker_fee):

        for order_id, order in self.orders_dict_latency.copy().items():
    
            #checking specific conditions
            if (order.side == 'BID' and order.price >=orderbook.asks[0] + maker_fee) or \
                (order.side == 'ASK' and order.price <= orderbook.bids[0] + maker_fee):
    
                own_trade_order = OwnTrade(trade.timestamp + (self.md_latency)*1000000, self.trade_id, order.order_id, order.size, order.price, order.side)
                self.own_trades[self.trade_id] = own_trade_order
                self.orders_dict_latency.pop(order_id)
                self.trade_id += 1
            
            elif (order.side == 'BID' and trade.side == 'ASK' and order.price >= trade.price + maker_fee) or \
                (order.side == 'ASK' and trade.side == 'BID' and order.price <= trade.price + maker_fee):
                
                own_trade_order = OwnTrade(trade.timestamp + (self.md_latency)*1000000, self.trade_id, order.order_id, order.size, order.price, order.side)
                self.own_trades[self.trade_id] = own_trade_order
                self.orders_dict_latency.pop(order_id)
                self.trade_id += 1
                
        return self.own_trades

    #this function creates an exemplar of the Order class
    def place_order(self, side, size, price, timestamp):
        order = Order(self.order_id, size, price, timestamp, side)
        self.order_id += 1
        self.orders_dict[self.order_id] = order

        return self.orders_dict

    #this function cancels order due to specific conditions
    def cancel_order(self, order_id):
        self.orders_dict_latency.pop(order_id)


if __name__ == "__main__":
    #create our strategy
    strategy = Strategy(0.001, 400, 0, [])
    #create out simulator
    sim = Sim(10, 10)
    #calculate our pnl
    pnl_data = strategy.run(sim)
    pnl_data = pd.DataFrame(pnl_data1)
    pnl_data.to_csv('pnl_data.csv')