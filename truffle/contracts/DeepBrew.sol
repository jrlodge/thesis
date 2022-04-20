pragma solidity ^0.8.13;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract DeepBrew is ERC20 {
    constructor(uint256 initialSupply) public ERC20("DeepBrew", "BEER") {
        _mint(msg.sender, initialSupply);
    }
}