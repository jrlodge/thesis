import gym
from gym import spaces
import numpy as np

import web3 as Web3
from ethtoken.abi import EIP20_ABI

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
        round(amount), # round because ERC20 can't be divided, thus can only be traded in integer quantities
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

class BeerGameEnv(gym.Env):
    def __init__(self):
        super(BeerGameEnv, self).__init__()
        self.action_space = spaces.Box(0,100_000)
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(6,), dtype=np.float32)
    
    def step(self, action):
        return observation, self.reward, self.done, info
    
    def reset(self):
        return observation