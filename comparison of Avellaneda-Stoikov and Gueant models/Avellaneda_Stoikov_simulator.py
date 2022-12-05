import pandas as pd
from collections import deque
from dataclasses import dataclass
#from math import gamma
#from signal import Sigmasks
from typing import List, Optional, Tuple, Union, Deque, Dict
import numpy as np
from sortedcontainers import SortedDict

@dataclass
class Order:  # Our own placed order
    place_ts : float # ts when we place the order
    exchange_ts : float # ts when exchange(simulator) get the order    
    order_id: int
    side: str
    size: float
    price: float

        
@dataclass
class CancelOrder:
    exchange_ts: float
    id_to_delete : int

@dataclass
class AnonTrade:  # Market trade
    exchange_ts : float
    receive_ts : float
    side: str
    size: float
    price: float


@dataclass
class OwnTrade:  # Execution of own placed order
    place_ts : float # ts when we call place_order method, for debugging
    exchange_ts: float
    receive_ts: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float
    execute : str # BOOK or TRADE


    def __post_init__(self):
        assert isinstance(self.side, str)

@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    exchange_ts : float
    receive_ts : float
    asks: List[Tuple[float, float]]  # tuple[price, size]
    bids: List[Tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    exchange_ts : float
    receive_ts : float
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trade: Optional[AnonTrade] = None


def update_best_positions(best_bid:float, best_ask:float, md:MdUpdate) -> Tuple[float, float]:
    if not md.orderbook is None:
        best_bid = md.orderbook.bids[0][0]
        best_ask = md.orderbook.asks[0][0]

    return best_bid, best_ask


class Sim:
    def __init__(self, market_data: List[MdUpdate], execution_latency: float, md_latency: float) -> None:
        '''
            Args:
                market_data(List[MdUpdate]): market data
                execution_latency(float): latency in nanoseconds
                md_latency(float): latency in nanoseconds
        '''   
        self.inventory = 0 #our current inventory

        #transform md to queue
        self.md_queue = deque( market_data )
        #action queue
        self.actions_queue:Deque[ Union[Order, CancelOrder] ] = deque()
        #SordetDict: receive_ts -> [updates]
        self.strategy_updates_queue = SortedDict()
        #map : order_id -> Order
        self.ready_to_execute_orders:Dict[int, Order] = {}
        
        #current md
        self.md:Optional[MdUpdate] = None
        #current ids
        self.order_id = 0
        self.trade_id = 0
        #latency
        self.latency = execution_latency
        self.md_latency = md_latency
        #current bid and ask
        self.best_bid = -np.inf
        self.best_ask = np.inf
        self.mid_price = 0
        #current trade 
        self.trade_price = {}
        self.trade_price['BID'] = -np.inf
        self.trade_price['ASK'] = np.inf
        #last order
        self.last_order:Optional[Order] = None
        
    
    def get_md_queue_event_time(self) -> float:
        return np.inf if len(self.md_queue) == 0 else self.md_queue[0].exchange_ts
    
    
    def get_actions_queue_event_time(self) -> float:
        return np.inf if len(self.actions_queue) == 0 else self.actions_queue[0].exchange_ts
    
    
    def get_strategy_updates_queue_event_time(self) -> float:
        return np.inf if len(self.strategy_updates_queue) == 0 else self.strategy_updates_queue.keys()[0]
    
    
    def get_order_id(self) -> int:
        res = self.order_id
        self.order_id += 1
        return res
    
    
    def get_trade_id(self) -> int:
        res = self.trade_id
        self.trade_id += 1
        return res
    

    def update_best_pos(self) -> None:
        assert not self.md is None, "no current market data!" 
        if not self.md.orderbook is None:
            self.best_bid = self.md.orderbook.bids[0][0]#МЭЭЭЭЭЭЭЭЭЭЭЭЭ
            self.best_ask = self.md.orderbook.asks[0][0]#"MUUUUUUUUUUUUUUU"
            #self.mid_price = (self.md.orderbook.bids[0][0] + self.md.orderbook.asks[0][0]) * 0.5
    
    
    def update_last_trade(self) -> None:
        assert not self.md is None, "no current market data!"
        if not self.md.trade is None:
            self.trade_price[self.md.trade.side] = self.md.trade.price


    def delete_last_trade(self) -> None:
        self.trade_price['BID'] = -np.inf
        self.trade_price['ASK'] = np.inf


    def update_md(self, md:MdUpdate) -> None:
        #current orderbook
        self.md = md 
        #update position
        self.update_best_pos()
        #update info about last trade
        self.update_last_trade()

        #add md to strategy_updates_queue
        if not md.receive_ts in self.strategy_updates_queue.keys():
            self.strategy_updates_queue[md.receive_ts] = []
        self.strategy_updates_queue[md.receive_ts].append(md)
        
    
    def update_action(self, action:Union[Order, CancelOrder]) -> None:
        
        if isinstance(action, Order):
            #self.ready_to_execute_orders[action.order_id] = action
            #save last order to try to execute it aggressively
            self.last_order = action
        elif isinstance(action, CancelOrder):    
            #cancel order
            if action.id_to_delete in self.ready_to_execute_orders:
                self.ready_to_execute_orders.pop(action.id_to_delete)
        else:
            assert False, "Wrong action type!"

        
    def tick(self) -> tuple([ float, float, List[ Union[OwnTrade, MdUpdate] ]]):
        '''
            Simulation tick

            Returns:
                receive_ts(float): receive timestamp in nanoseconds
                res(List[Union[OwnTrade, MdUpdate]]): simulation result. 
        '''
        while True:     
            #get event time for all the queues
            strategy_updates_queue_et = self.get_strategy_updates_queue_event_time()
            md_queue_et = self.get_md_queue_event_time()
            actions_queue_et = self.get_actions_queue_event_time()
            
            #if both queue are empty
            if md_queue_et == np.inf and actions_queue_et == np.inf:
                break

            #strategy queue has minimum event time
            if strategy_updates_queue_et < min(md_queue_et, actions_queue_et):
                break


            if md_queue_et <= actions_queue_et:
                self.update_md( self.md_queue.popleft() )
            if actions_queue_et <= md_queue_et:
                self.update_action( self.actions_queue.popleft() )

            #execute last order aggressively
            self.execute_last_order()
            #execute orders with current orderbook
            self.execute_orders()
            #delete last trade
            self.delete_last_trade()
        #end of simulation
        if len(self.strategy_updates_queue) == 0:
            return self.inventory, np.inf, None
        key = self.strategy_updates_queue.keys()[0]
        res = self.strategy_updates_queue.pop(key)

        #print( self.inventory, key, res)
        return self.inventory, key, res

    def execute_last_order(self) -> None:
        '''
            this function tries to execute self.last order aggressively
        '''
        #nothing to execute
        if self.last_order is None:
            return

        executed_price, execute = None, None
        #
        if self.last_order.side == 'BID' and self.last_order.price >= self.best_ask:
            executed_price = self.best_ask
            execute = 'BOOK'
        #    
        elif self.last_order.side == 'ASK' and self.last_order.price <= self.best_bid:
            executed_price = self.best_bid
            execute = 'BOOK'

        if not executed_price is None:
            executed_order = OwnTrade(
                self.last_order.place_ts, # when we place the order
                self.md.exchange_ts, #exchange ts
                self.md.exchange_ts + self.md_latency, #receive ts
                self.get_trade_id(), #trade id
                self.last_order.order_id, 
                self.last_order.side, 
                self.last_order.size, 
                executed_price, execute)
            #add order to strategy update queue
            #there is no defaultsorteddict so I have to do this
            if not executed_order.receive_ts in self.strategy_updates_queue:
                self.strategy_updates_queue[ executed_order.receive_ts ] = []
            self.strategy_updates_queue[ executed_order.receive_ts ].append(executed_order)
        else:
            self.ready_to_execute_orders[self.last_order.order_id] = self.last_order

        #delete last order
        self.last_order = None


    def execute_orders(self) -> None:
        executed_orders_id = []
        for order_id, order in self.ready_to_execute_orders.items():

            executed_price, execute = None, None

            #
            if order.side == 'BID' and order.price >= self.best_ask:
                executed_price = order.price
                execute = 'BOOK'
            #    
            elif order.side == 'ASK' and order.price <= self.best_bid:
                executed_price = order.price
                execute = 'BOOK'
            #
            elif order.side == 'BID' and order.price >= self.trade_price['ASK']:
                executed_price = order.price
                execute = 'TRADE'    
            #
            elif order.side == 'ASK' and order.price <= self.trade_price['BID']:
                executed_price = order.price
                execute = 'TRADE'

            if not executed_price is None:
                executed_order = OwnTrade(
                    order.place_ts, # when we place the order
                    self.md.exchange_ts, #exchange ts
                    self.md.exchange_ts + self.md_latency, #receive ts
                    self.get_trade_id(), #trade id
                    order_id, order.side, order.size, executed_price, execute)
        
                executed_orders_id.append(order_id)

                #added order to strategy update queue
                #there is no defaultsorteddict so i have to do this
                if not executed_order.receive_ts in self.strategy_updates_queue:
                    self.strategy_updates_queue[ executed_order.receive_ts ] = []
                self.strategy_updates_queue[ executed_order.receive_ts ].append(executed_order)
                if executed_order.side == 'BID':
                    self.inventory += executed_order.size
                elif executed_order.side == 'ASK':
                    self.inventory -= executed_order.size
        
        #deleting executed orders
        for k in executed_orders_id:
            self.ready_to_execute_orders.pop(k)


    def place_order(self, ts:float, size:float, side:str, price:float) -> Order:
        #добавляем заявку в список всех заявок
        order = Order(ts, ts + self.latency, self.get_order_id(), side, size, price)
        self.actions_queue.append(order)
        return order

    
    def cancel_order(self, ts:float, id_to_delete:int) -> CancelOrder:
        #добавляем заявку на удаление
        ts += self.latency
        delete_order = CancelOrder(ts, id_to_delete)
        self.actions_queue.append(delete_order)
        return delete_order
    
#-------------------STRATEGY-------------------------
class BestPosStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is  canceled.
    '''
    def __init__(self, delay: float, sigma: float, gamma: float, k: float, hold_time:Optional[float]) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                sigma: volatility
                gamma: attitude to risk
                k: arrival rate
                hold_time(Optional[float]): holding time in nanoseconds
                
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time

        self.sigma = sigma #volatility
        self.gamma = gamma #attitude to risk
        self.k = k #arrival rate of orders as the change in volume per second
        #we use it to normalize time
        self.T_begin = 1655953200999785662 #beginning of our session
                       
        self.T_end = 1655974797846439628	 #end of our session
        self.q = []
        self.spread = []


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            inventory, receive_ts, updates = sim.tick()
            self.q.append([inventory, receive_ts])
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                #calculate the mid-price
                mid = 0.5 * (best_bid + best_ask)
                #normalize time
                T_t = 1 - (receive_ts - self.T_begin)/(self.T_end - self.T_begin)
                #calculate difference between reservation price and mid price
                mid_reservation_spread = self.gamma * self.sigma * self.sigma * inventory * T_t
                #calculate agent's spread  
                spread = self.gamma * self.sigma * self.sigma * T_t + (2/self.gamma) * np.log(1 + (self.gamma/self.k))
                #calculate current ask/bid prices
                ask_price = mid - mid_reservation_spread + 0.5 * spread
                bid_price = mid - mid_reservation_spread - 0.5 * spread
                self.spread.append([ask_price, bid_price, receive_ts])
                bid_order = sim.place_order( receive_ts, 0.001, 'BID', bid_price ) #подредачить
                ask_order = sim.place_order( receive_ts, 0.001, 'ASK', ask_price ) #подредачить
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders, self.q, self.spread
#--------------------------------------------------------LOADDATA----------------------------
    
    
def load_trades(path, nrows=10000) -> List[AnonTrade]:
    '''
        This function downloads trades data

        Args:
            path(str): path to file
            nrows(int): number of rows to read

        Return:
            trades(List[AnonTrade]): list of trades 
    '''
    trades = pd.read_csv(path + 'trades.csv', nrows=nrows)
    trades = trades.loc[719000:1835700, :]
    #переставляю колонки, чтобы удобнее подавать их в конструктор AnonTrade
    trades = trades[ ['exchange_ts', 'receive_ts', 'aggro_side', 'size', 'price' ] ].sort_values(["exchange_ts", 'receive_ts'])
    receive_ts = trades.receive_ts.values
    exchange_ts = trades.exchange_ts.values 
    trades = [ AnonTrade(*args) for args in trades.values]
    return trades


def load_books(path, nrows=10000) -> List[OrderbookSnapshotUpdate]:
    '''
        This function downloads orderbook market data

        Args:
            path(str): path to file
            nrows(int): number of rows to read

        Return:
            books(List[OrderbookSnapshotUpdate]): list of orderbooks snapshots 
    '''
    lobs   = pd.read_csv(path + 'lobs.csv', nrows=nrows)
    lobs = lobs.loc[314000:957000, :]
    #lobs = lobs.iloc[:10000, :]
    #rename columns
    names = lobs.columns.values
    ln = len('btcusdt:Binance:LinearPerpetual_')
    renamer = { name:name[ln:] for name in names[2:]}
    renamer[' exchange_ts'] = 'exchange_ts'
    lobs.rename(renamer, axis=1, inplace=True)
    
    #timestamps
    receive_ts = lobs.receive_ts.values
    exchange_ts = lobs.exchange_ts.values 
    #список ask_price, ask_vol для разных уровней стакана
    #размеры: len(asks) = 10, len(asks[0]) = len(lobs)
    asks = [list(zip(lobs[f"ask_price_{i}"],lobs[f"ask_vol_{i}"])) for i in range(10)]
    #транспонируем список
    asks = [ [asks[i][j] for i in range(len(asks))] for j in range(len(asks[0]))]
    #тоже самое с бидами
    bids = [list(zip(lobs[f"bid_price_{i}"],lobs[f"bid_vol_{i}"])) for i in range(10)]
    bids = [ [bids[i][j] for i in range(len(bids))] for j in range(len(bids[0]))]
    
    books = list( OrderbookSnapshotUpdate(*args) for args in zip(exchange_ts, receive_ts, asks, bids) )
    return books


def merge_books_and_trades(books : List[OrderbookSnapshotUpdate], trades: List[AnonTrade]) -> List[MdUpdate]:
    '''
        This function merges lists of orderbook snapshots and trades 
    '''
    trades_dict = { (trade.exchange_ts, trade.receive_ts) : trade for trade in trades }
    books_dict  = { (book.exchange_ts, book.receive_ts) : book for book in books }
    
    ts = sorted(trades_dict.keys() | books_dict.keys())

    md = [MdUpdate(*key, books_dict.get(key, None), trades_dict.get(key, None)) for key in ts]
    return md


def load_md_from_file(path: str, nrows=10000) -> List[MdUpdate]:
    '''
        This function downloads orderbooks ans trades and merges them
    '''
    books  = load_books(path, nrows)
    trades = load_trades(path, nrows)
    return merge_books_and_trades(books, trades)

#------------------------------GETPNL---------------------------

def get_pnl(updates_list:List[ Union[MdUpdate, OwnTrade] ]) -> pd.DataFrame:
    '''
        This function calculates PnL from list of updates
    '''

    #current position in btc and usd
    btc_pos, usd_pos = 0.0, 0.0
    
    N = len(updates_list)
    btc_pos_arr = np.zeros((N, ))
    usd_pos_arr = np.zeros((N, ))
    mid_price_arr = np.zeros((N, ))
    #current best_bid and best_ask
    best_bid:float = -np.inf
    best_ask:float = np.inf

    for i, update in enumerate(updates_list):
        
        if isinstance(update, MdUpdate):
            best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
        #mid price
        #i use it to calculate current portfolio value
        mid_price = 0.5 * ( best_ask + best_bid )
        
        if isinstance(update, OwnTrade):
            trade = update    
            #update positions
            if trade.side == 'BID':
                btc_pos += trade.size
                usd_pos -= trade.price * trade.size
            elif trade.side == 'ASK':
                btc_pos -= trade.size
                usd_pos += trade.price * trade.size
        #current portfolio value
        
        btc_pos_arr[i] = btc_pos
        usd_pos_arr[i] = usd_pos
        mid_price_arr[i] = mid_price
    
    worth_arr = btc_pos_arr * mid_price_arr + usd_pos_arr
    receive_ts = [update.receive_ts for update in updates_list]
    exchange_ts = [update.exchange_ts for update in updates_list]
    
    df = pd.DataFrame({"exchange_ts": exchange_ts, "receive_ts":receive_ts, "total":worth_arr, "BTC":btc_pos_arr, 
                       "USD":usd_pos_arr, "mid_price":mid_price_arr})
    df = df.groupby('receive_ts').agg(lambda x: x.iloc[-1]).reset_index()    
    return df


def trade_to_dataframe(trades_list:List[OwnTrade]) -> pd.DataFrame:
    exchange_ts = [ trade.exchange_ts for trade in trades_list ]
    receive_ts = [ trade.receive_ts for trade in trades_list ]
    
    size = [ trade.size for trade in trades_list ]
    price = [ trade.price for trade in trades_list ]
    side  = [trade.side for trade in trades_list ]
    
    dct = {
        "exchange_ts" : exchange_ts,
        "receive_ts"  : receive_ts,
         "size" : size,
        "price" : price,
        "side"  : side
    }

    df = pd.DataFrame(dct).groupby('receive_ts').agg(lambda x: x.iloc[-1]).reset_index()    
    return df


def md_to_dataframe(md_list: List[MdUpdate]) -> pd.DataFrame:
    
    best_bid = -np.inf
    best_ask = np.inf
    best_bids = []
    best_asks = []
    for md in md_list:
        best_bid, best_ask = update_best_positions(best_bid, best_ask, md)
        best_bids.append(best_bid)
        best_asks.append(best_ask)
        
    exchange_ts = [ md.exchange_ts for md in md_list ]
    receive_ts = [ md.receive_ts for md in md_list ]
    dct = {
        "exchange_ts" : exchange_ts,
        "receive_ts"  :receive_ts,
        "bid_price" : best_bids,
        "ask_price" : best_asks
    }
    
    df = pd.DataFrame(dct).groupby('receive_ts').agg(lambda x: x.iloc[-1]).reset_index()    
    return df

if __name__ == "__main__":
    PATH_TO_FILE = 'D:/it/cmf/hft/week1/md/md/btcusdt_Binance_LinearPerpetual/'
    md = load_md_from_file(PATH_TO_FILE, nrows=2000000)
    latency = pd.Timedelta(10, 'ms').delta
    md_latency = pd.Timedelta(10, 'ms').delta
    sim = Sim(md, latency, md_latency)
    delay = pd.Timedelta(0.1, 's').delta
    hold_time = pd.Timedelta(10, 's').delta
    sigma, gamma, k = 10.0, 0.1, 1.0
    strategy = BestPosStrategy(delay, sigma, gamma, k, hold_time)
    trades_list, md_list, updates_list, all_orders, q, spread = strategy.run(sim)
    pnl = get_pnl(updates_list)
    pnl = pd.DataFrame(pnl)
    spread = pd.DataFrame(spread)
    q = pd.DataFrame(q)
    spread.to_csv('total_spread_stoikov.csv')
    q.to_csv('total_q_stoikov.csv')
    pnl.to_csv('total_pnl_stoikov.csv')
   