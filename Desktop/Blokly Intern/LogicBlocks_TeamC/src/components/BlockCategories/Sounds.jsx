// BlocklyBlocksAndCategory.js
import Blockly from "blockly";

// Define the 'play_sound' block
Blockly.Blocks["play_sound"] = {
  init: function () {
    this.appendDummyInput().appendField("Play Sound");
    this.appendDummyInput()
      .appendField("Sound Name:")
      .appendField(new Blockly.FieldTextInput("meow"), "SOUND_NAME");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Play a sound");
  },
  // Generator code for 'play_sound' block
  generateCode: function (block) {
    var soundName = block.getFieldValue("SOUND_NAME");
    return `playSound("${soundName}");\n`;
  },
};

// Define the 'start_sound' block
Blockly.Blocks["start_sound"] = {
  init: function () {
    this.appendDummyInput().appendField("Start Sound");
    this.appendDummyInput()
      .appendField("Sound Name:")
      .appendField(new Blockly.FieldTextInput("meow"), "SOUND_NAME");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Start playing a sound");
  },
  // Generator code for 'start_sound' block
  generateCode: function (block) {
    var soundName = block.getFieldValue("SOUND_NAME");
    return `startSound("${soundName}");\n`;
  },
};

// Define the 'change_by_effect' block
Blockly.Blocks["change_by_effect"] = {
  init: function () {
    this.appendDummyInput().appendField("Change Sound by Effect");
    this.appendDummyInput()
      .appendField("Effect Type:")
      .appendField(new Blockly.FieldTextInput("pitch"), "EFFECT_TYPE");
    this.appendDummyInput()
      .appendField("Amount:")
      .appendField(new Blockly.FieldNumber(10), "AMOUNT");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Change the sound by a specified effect");
  },
  // Generator code for 'change_by_effect' block
  generateCode: function (block) {
    var effectType = block.getFieldValue("EFFECT_TYPE");
    var amount = block.getFieldValue("AMOUNT");
    return `changeSoundByEffect("${effectType}", ${amount});\n`;
  },
};

// Define the 'set_by_effect' block
Blockly.Blocks["set_by_effect"] = {
  init: function () {
    this.appendDummyInput().appendField("Set Sound by Effect");
    this.appendDummyInput()
      .appendField("Effect Type:")
      .appendField(new Blockly.FieldTextInput("pitch"), "EFFECT_TYPE");
    this.appendDummyInput()
      .appendField("Value:")
      .appendField(new Blockly.FieldNumber(50), "VALUE");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Set the sound by a specified effect");
  },
  // Generator code for 'set_by_effect' block
  generateCode: function (block) {
    var effectType = block.getFieldValue("EFFECT_TYPE");
    var value = block.getFieldValue("VALUE");
    return `setSoundByEffect("${effectType}", ${value});\n`;
  },
};

// Define the 'clear_sound_effects' block
Blockly.Blocks["clear_sound_effects"] = {
  init: function () {
    this.appendDummyInput().appendField("Clear Sound Effects");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Clear all sound effects");
  },
  // Generator code for 'clear_sound_effects' block
  generateCode: function (block) {
    return "clearSoundEffects();\n";
  },
};

// Define the 'change_volume_by' block
Blockly.Blocks["change_volume_by"] = {
  init: function () {
    this.appendDummyInput().appendField("Change Volume by");
    this.appendDummyInput()
      .appendField("Amount:")
      .appendField(new Blockly.FieldNumber(10), "AMOUNT");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Change the volume by a specified amount");
  },
  // Generator code for 'change_volume_by' block
  generateCode: function (block) {
    var amount = block.getFieldValue("AMOUNT");
    return `changeVolumeBy(${amount});\n`;
  },
};

// Define the 'set_volume_to' block
Blockly.Blocks["set_volume_to"] = {
  init: function () {
    this.appendDummyInput().appendField("Set Volume to");
    this.appendDummyInput()
      .appendField("Volume:")
      .appendField(new Blockly.FieldNumber(50), "VOLUME");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Set the volume to a specified value");
  },
  // Generator code for 'set_volume_to' block
  generateCode: function (block) {
    var volume = block.getFieldValue("VOLUME");
    return `setVolumeTo(${volume});\n`;
  },
};

// Category definition
export const Sounds = `
  <category name="Sounds" colour="">
    <block type="play_sound"></block>
    <block type="start_sound"></block>
    <block type="change_by_effect"></block>
    <block type="set_by_effect"></block>
    <block type="clear_sound_effects"></block>
    <block type="change_volume_by"></block>
    <block type="set_volume_to"></block>
  </category>
`;
