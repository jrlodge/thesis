import gym
from gym import spaces
import numpy as np
# import web3 as Web3
from the_beer_game import web3
from the_beer_game import Web3
from ethtoken.abi import EIP20_ABI

STARTING_INVENTORY = 12
STARTING_BALANCE = 200_000
STARTING_DEMAND = 4
ROUNDS = 10
BEER_PRICE = 0.002

# connect to ganache
ganache_url = 'HTTP://127.0.0.1:8545'
web3 = Web3(Web3.HTTPProvider(ganache_url))

# accounts dictionary
# mnemonic: myth like bonus scare over problem client lizard pioneer submit female collect
accounts = {
    'manufacturer': {
        'address': '0xFE41FE950d4835bD539AC24fBaaDED16b2E32922',
        'private_key': '0x4d8631d58af474e97e8646472077626207ed5c32a0a956b256811401e756f1a3'
                    },
    'distributor':  {
        'address': '0x45928E9F64590F28c964E1d73a01Ad0311896b4B',
        'private_key': '0x4836c020dedd3e96f1ebfdb55986e1d7aeac5bf26bf154550da87a1e5e34049d'
                    },
    'wholesaler': {
        'address': '0x7c66F5C9e97aC36B181BFE17CFB6303EE32C270e',
        'private_key': '0xd189172812b452424cc60aa57f6c6321c3f552ac45dedb0a6baa20419963326e'
    },
    'retailer': {
        'address': '0xc1ba023e51396C0e9891026736BcaB4ecfB587E3',
        'private_key': '0x103cafa8c177c0eb9eeee852fc3e3b24a0869a59a91df78527abeef9f719eba7'        
    },
    'market': {
        'address': '0x6CDf8da55Ba3b54D52d1F21392cc2409b1153E88',
        'private_key': '0xce4b3f8a6364c266b2a155acc16b379451e50c5e54eccfb119474943447a9f30'
    }
}
# BEER functions
# ERC20 contract address (DeepBrew, BEER)
# need to find a way to grab the contract address with code rather than manually
deepbrew_contract = Web3.toChecksumAddress('0x6dF43d5EFD4ddE3cC72EDf36F012A5c390b628aC')

# ERC20 contract object
deepbrew = web3.eth.contract(address=deepbrew_contract, abi=EIP20_ABI)

# returns the BEER balance of a given account
# INPUTS: account NAME, as a string, e.g. 'manufacturer'
def get_inventory(account):
    return float(deepbrew.functions.balanceOf(accounts[account]['address']).call())

'''
used to send BEER from one wallet to another
INPUTS
from_account: string, name of the account sending
to_account: string, name of the account receiving
amount: integer, amount to be sent in BEER
'''
def send_beer(from_account, to_account, amount):
    if amount == 0:
        return
    nonce = web3.eth.getTransactionCount(accounts[from_account]['address'])

    deepbrew_txn = deepbrew.functions.transfer(
        accounts[to_account]['address'],
        round(min(get_inventory(from_account), amount)), # round because ERC20 can't be divided, thus can only be traded in integer quantities
    ).buildTransaction({
        'gas': 70000,
        'gasPrice': web3.toWei('50', 'gwei'),
        'nonce': nonce,
    })

    signed_txn = web3.eth.account.signTransaction(deepbrew_txn, accounts[from_account]['private_key'])
    signed_txn.hash
    signed_txn.rawTransaction
    signed_txn.r
    signed_txn.s
    signed_txn.v
    web3.eth.sendRawTransaction(signed_txn.rawTransaction) 
    web3.toHex(web3.sha3(signed_txn.rawTransaction))

'''
function called at the start of each round to return agent inventory to initial conditions
INPUTS
balance: the desired starting inventory for each agent in BEER
'''
def reset_inventories(inventory):
    # send all current inventory to the market
    for account in accounts:
        if account != 'market' and deepbrew.functions.balanceOf(accounts[account]['address']).call() != 0:
            send_beer(account, 'market', deepbrew.functions.balanceOf(accounts[account]['address']).call())
    for account in accounts:
        if account != 'market':
            send_beer('market', account, inventory)

# ETH functions

# returns the ETH balance of a given account
# INPUTS: account NAME, as a string, e.g. 'manufacturer'
def get_balance(account):
    return float(web3.fromWei(web3.eth.getBalance(accounts[account]['address']), 'ether'))

'''
used to send eth from one wallet to another
INPUTS
from_account: string, name of the account sending
to_account: string, name of the account receiving
amount: integer, amount to be sent in ETH
'''
def send_eth(from_account, to_account, amount):
    if amount == 0:
        return
    # calculate nonce from sender account
    nonce = web3.eth.getTransactionCount(accounts[from_account]['address'])
    #build tx
    tx = {
        'nonce': nonce,
        'to': accounts[to_account]['address'],
        'value': web3.toWei(amount, 'ether'),
        'gas': 200000,
        'gasPrice': web3.toWei('50', 'gwei')
    }
    # sign the transaction
    singed_tx = web3.eth.account.signTransaction(tx, accounts[from_account]['private_key'])
    # send transaction
    tx_hash = web3.eth.sendRawTransaction(singed_tx.rawTransaction)
    # get transaction hash
    # print("Transaction hash: ", web3.toHex(tx_hash))
    # print latest block number
    # print("Block numbers: ", web3.eth.block_number)

'''
function called at the start of each round to return agent balances to initial conditions
INPUTS
balance: the desired starting balance for each agent in ETH
'''
def reset_balances(balance):
    # send all current balances to the market
    for account in accounts:
        if account != 'market':
            # if the agent has more than 1 ETH, send all but 1 ETH to market to cover gas
            if web3.fromWei(web3.eth.getBalance(accounts[account]['address']), 'ether') >= 1: 
                send_eth(account, 'market', (web3.fromWei(web3.eth.getBalance(accounts[account]['address']), 'ether')-1))
            # if the agent has less than 1 ETH, the market sends them the amount such that they now have 1 ETH
            else:
                send_eth('market', account, 1-(web3.fromWei(web3.eth.getBalance(accounts[account]['address']), 'ether')))
                
    # redistribute desired ETH to each agent
    for account in accounts:
        if account != 'market' and balance != 1: # need to leave behind 1 ETH to cover gas fees, in the event that desired balance is 1, send nothing back
            send_eth('market', account, balance-1) # send back the desired amount -1 


'''
ENVIRONMENT CLASS
'''
class BeerGameEnv(gym.Env):
    def __init__(self):
        super(BeerGameEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=5_000,shape=(1,)) # range of actions that can be taken 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, # supposedly low and high are fairly arbitrary
                                            shape=(8,), dtype=np.float64) # not sure about shape or dtype
    
    def step(self, action):
        
        reset_balances(STARTING_BALANCE)
        reset_inventories(STARTING_INVENTORY)
        
        for i in range(ROUNDS):
            # increase demand 
            if i == 0:
                self.demand.append(STARTING_DEMAND)
                print('You are playing as the distributor.')
            #elif i % 10 == 0:
                #demand.append(round(demand[i-1]*1.5)) # increase demand by 50% every 10th round
            else:
                self.demand.append(self.demand[i-1]*1.1) # increase demand by 10% every other round
            
            print('Round', i+1, 'of', ROUNDS)
            
            # send deliveries 
            send_beer('market', 'manufacturer', self.deliveries_to_manufacturer[i])
            send_beer('manufacturer', 'distributor', self.deliveries_to_distributor[i])
            send_beer('distributor', 'wholesaler', self.deliveries_to_wholesaler[i])
            send_beer('wholesaler', 'retailer', self.deliveries_to_retailer[i])
            send_beer('retailer', 'market', self.deliveries_to_market[i])

            # store inventories for this i (after deliveries)
            self.manufacturer_inventory.append(get_inventory('manufacturer'))
            self.distributor_inventory.append(get_inventory('distributor'))
            self.wholesaler_inventory.append(get_inventory('wholesaler'))
            self.retailer_inventory.append(get_inventory('retailer'))
            self.market_inventory.append(get_inventory('market'))

            # store balances for this i
            self.manufacturer_balance.append(get_balance('manufacturer'))
            self.distributor_balance.append(get_balance('distributor'))
            self.wholesaler_balance.append(get_balance('wholesaler'))
            self.retailer_balance.append(get_balance('retailer'))
            self.market_balance.append(get_balance('market'))

            # calculate and store backorders
            self.retailer_backorder.append(round(sum(self.orders_from_market) - sum(self.deliveries_to_market)))
            self.wholesaler_backorder.append(round(sum(self.orders_from_retailer) - sum(self.deliveries_to_retailer)))
            self.distributor_backorder.append(round(sum(self.orders_from_wholesaler) - sum(self.deliveries_to_wholesaler)))
            self.manufacturer_backorder.append(round(sum(self.orders_from_distributor) - sum(self.deliveries_to_distributor)))

            # calculate base-stock: average demand from last 4 rounds + safety stock (the forecasted next market demand: last demand * average increase of last 4 rounds)
            safety_stock = 1.1*STARTING_DEMAND
            if i < 3: 
                self.base_stock.append(4*np.average(self.demand)+safety_stock)
            else:
                self.base_stock.append(4*np.average(self.demand[-4:])+safety_stock)
            # calculate inventory position
            if i == 0:
                self.retailer_position.append(self.retailer_inventory[0])
                self.wholesaler_position.append(self.wholesaler_inventory[0])
                self.distributor_position.append(self.distributor_inventory[0])
                self.manufacturer_position.append(self.manufacturer_inventory[0])
            else:
                '''
                # my equation
                retailer_position.append(round(retailer_inventory[i] + orders_from_retailer[i-1]
                                            + wholesaler_backorder[i] - orders_from_market[i-1] - retailer_backorder[i]))
                wholesaler_position.append(round(wholesaler_inventory[i] + orders_from_wholesaler[i-1]
                                                + distributor_backorder[i-1] - orders_from_retailer[i-1] - wholesaler_backorder[i]))
                distributor_position.append(round(distributor_inventory[i] + orders_from_distributor[i-1]
                                                + manufacturer_backorder[i] - orders_from_wholesaler[i-1] - distributor_backorder[i]))
                manufacturer_position.append(round(manufacturer_inventory[i] + orders_from_manufacturer[i-1]
                                                - orders_from_distributor[i-1] - manufacturer_backorder[i])) # manufacturer's supplier (market) has no backorder
                '''
                # flowlity's equation 
                self.retailer_position.append(round(sum(self.orders_from_retailer[-3:]) 
                                            + sum(self.wholesaler_backorder[-4:]) - self.retailer_backorder[i]))
                self.wholesaler_position.append(round(sum(self.orders_from_wholesaler[-3:]) 
                                                + sum(self.distributor_backorder[-4:]) - self.wholesaler_backorder[i]))
                self.distributor_position.append(round(sum(self.orders_from_distributor[-3:]) 
                                                + sum(self.manufacturer_backorder[-4:]) - self.distributor_backorder[i]))
                self.manufacturer_position.append(round(sum(self.orders_from_manufacturer[-3:]) 
                                                + sum(self.orders_from_distributor[-4:]) - self.manufacturer_backorder[i]))
                

            # calculate replenishment orders (base-stock policy), order zero if inventory position is higher than the base-stock
            self.orders_from_market.append(self.demand[i]) # market demand order from retailer
            self.orders_from_retailer.append(max(0,round(self.base_stock[i] - self.retailer_position[i])))
            self.orders_from_wholesaler.append(max(0,round(self.base_stock[i] - self.wholesaler_position[i])))

            self.orders_from_distributor.append(int(action))
            
            self.orders_from_manufacturer.append(max(0,round(self.base_stock[i] - self.manufacturer_position[i])))

            # send ETH corresponding to orders placed, a 50% markup is applied for each touchpoint in the supply chain
            send_eth('market', 'retailer', self.orders_from_market[i]*BEER_PRICE*3) # consumers purchase from retailer
            send_eth('retailer', 'wholesaler', self.orders_from_retailer[i]*BEER_PRICE*2.5)
            send_eth('wholesaler', 'distributor', self.orders_from_wholesaler[i]*BEER_PRICE*2)
            send_eth('distributor', 'manufacturer', self.orders_from_distributor[i]*BEER_PRICE*1.5)
            send_eth('manufacturer', 'market', self.orders_from_manufacturer[i]*BEER_PRICE) # cost to manufacture beer

            # calculate and append expenses
            self.retailer_expenses.append(get_balance('retailer')-self.retailer_balance[i])
            self.wholesaler_expenses.append(get_balance('wholesaler')-self.wholesaler_balance[i])
            self.distributor_expenses.append(get_balance('distributor')-self.distributor_balance[i])
            self.manufacturer_expenses.append(get_balance('manufacturer')-self.manufacturer_balance[i])

            # calculate and append deliveries for next round
            self.deliveries_to_manufacturer.append(round(min(get_inventory('market'), self.orders_from_manufacturer[i])))
            self.deliveries_to_distributor.append(round(min(get_inventory('manufacturer'), self.orders_from_distributor[i] + self.manufacturer_backorder[i])))
            self.deliveries_to_wholesaler.append(round(min(get_inventory('distributor'), self.orders_from_wholesaler[i] + self.distributor_backorder[i])))
            self.deliveries_to_retailer.append(round(min(get_inventory('wholesaler'), self.orders_from_retailer[i] + self.wholesaler_backorder[i])))
            self.deliveries_to_market.append(round(min(get_inventory('retailer'), self.orders_from_market[i] + self.retailer_backorder[i])))

            for account in accounts:
                print(account, get_balance(account), 'ETH')
                print(account, get_inventory(account), 'BEER')

            print('--------------------------------')

            if i == ROUNDS-1:
                print('Game complete.')
                
            # rewards (these will probably need a fair bit of tweaking)
            # disincentivize inventory less than 0
            if self.distributor_inventory[i] < 0:
                self.reward -= 100
            # disincentivize any backorder
            if self.distributor_backorder[i] > 0:
                self.reward -= 100
            # disincentivize ordering less than requested by client
            if self.orders_from_wholesaler[i] < self.deliveries_to_wholesaler[i]:
                self.reward -= 100
            # incentivize profits
            if self.distributor_balance[i] > self.distributor_balance[0]:
                self.reward += 100
            # disincentivize holding significantly more inventory than neighbours
            if self.distributor_inventory[i] > 2*self.manufacturer_inventory[i] or self.distributor_inventory[i] > 2*self.wholesaler_inventory[i]:
                self.reward -= 100
            # need to initialize reward variable in the event that none of the above happens
            else:
                self.reward = 0

        self.done = True
        
        info = {}
        
        observation = [self.distributor_inventory, self.distributor_balance, self.orders_from_wholesaler,
                       self.orders_from_distributor, self.distributor_backorder, self.manufacturer_backorder,
                       self.deliveries_to_distributor, self.deliveries_to_wholesaler]
        
        observation = np.array(observation)
        
        return observation, self.reward, self.done, info
    
    def reset(self):
        
        # reset balances and inventories
        reset_balances(STARTING_BALANCE)
        reset_inventories(STARTING_INVENTORY)
        # VARIABLES
        # demand
        self.demand = []
        # base stock
        self.base_stock = []
        # inventories
        self.manufacturer_inventory = []
        self.distributor_inventory = []
        self.wholesaler_inventory = []
        self.retailer_inventory = []
        self.market_inventory = []
        # balances
        self.manufacturer_balance = []
        self.distributor_balance = []
        self.wholesaler_balance = []
        self.retailer_balance = []
        self.market_balance = []
        # deliveries - offset index by 1 as deliveries are calculated for the turn after the current turn
        self.deliveries_to_manufacturer = [0]
        self.deliveries_to_distributor = [0]
        self.deliveries_to_wholesaler = [0]
        self.deliveries_to_retailer = [0]
        self.deliveries_to_market = [0]
        # orders 
        self.orders_from_market = []
        self.orders_from_retailer = []
        self.orders_from_wholesaler = []
        self.orders_from_distributor = []
        self.orders_from_manufacturer = []
        # inventory positions
        self.retailer_position = []
        self.wholesaler_position = []
        self.distributor_position = []
        self.manufacturer_position = []
        # backorder
        self.retailer_backorder = []
        self.wholesaler_backorder = []
        self.distributor_backorder = []
        self.manufacturer_backorder = []
        # expenses
        self.retailer_expenses = []
        self.wholesaler_expenses = []
        self.distributor_expenses = []
        self.manufacturer_expenses = []
        
        self.done = False
        
        observation = [self.distributor_inventory, self.distributor_balance, self.orders_from_wholesaler,
                       self.orders_from_distributor, self.distributor_backorder, self.manufacturer_backorder,
                       self.deliveries_to_distributor, self.deliveries_to_wholesaler]
        
        observation = np.array(observation)
        
        return observation
    
    '''# the_beer_game(starting_balance, starting_inventory, beer_price, starting_demand, rounds)
    df = the_beer_game(250000, 12, 0.002, 4, 50)'''