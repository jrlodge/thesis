# need to set up a proper project directory
TERMINAL 1
cd..
cd..
cd projects
cd the_beer_game
ganache-cli -d -m myth like bonus scare over problem client lizard pioneer submit female collect -a 5 -e 10000000 -l 10000000 --db ./ganache_db

# REMEMBER TO USE kudos
TERMINAL 2
cd..
cd..
cd projects
cd the_beer_game
cd truffle
truffle compile
truffle migrate
# may need to wait a few seconds to load
truffle console
const deepbrew = await DeepBrew.deployed()
(await deepbrew.totalSupply()).toString()
(await deepbrew.balanceOf('0xFE41FE950d4835bD539AC24fBaaDED16b2E32922')).toString()