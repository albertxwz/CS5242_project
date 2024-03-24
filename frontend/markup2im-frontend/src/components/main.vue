<template>
  <div class="main-window">
    <div class="title">
      <span id="title-text">Markup to Image Demo</span>
      <span id="author-text"><i style="float: right; margin-right: 5%;">Zhang Xuan, Xie Wenzheng, Chen Feitong, Liu Zhicheng</i></span>
    </div>
    <div class="function-area">
      <div class="input-window">
        <div id="selector">
          <span style="float: left; margin-right: 2%; font-size: 200%; font-family: trebuchet ms; color: white;">Model: </span>
          <Dropdown style="width: 25%; float: left;" v-model="selectedModel" :options="models" optionValue="value" optionLabel="name" placeholder="Select a Model"/>
        </div>
        <div id="editor">
          <CodeEditor 
            :wrap="true"
            :modelValue.sync="code"
            @content="getContent"
            theme="gradient-dark" 
            :languages="[['latex', 'LaTex'],['html', 'HTML'],['lilypond', 'LilyPond'], ['smiles', 'SMILES']]"
            width="100%" 
            height="100%">
          </CodeEditor>
        </div>

        <div style="width: 100%; height: 10%;">
          <button id="generate-button" @click="GenerateImage()">Generate</button>
        </div>
      </div>
      <div class="output-window">
        <div class="generating">
          <span class="generate_text" v-if="start && !finish" style="float: left;">Generate Step: {{ step }}</span>
          <span class="generate_text" v-else-if="!start && !finish" style="float: left;">Press to Generate</span>
          <span class="generate_text" v-else-if="finish" style="float: left;">Generate Successfully!</span>
          <ProgressBar v-if="start && !finish" style="height: 4%; width: 100%;" mode="indeterminate"></ProgressBar>
          <ProgressBar v-else-if="!start && !finish" style="height: 4%; width: 100%;" :value="0"></ProgressBar>
          <ProgressBar v-else-if="finish" style="height: 4%; width: 100%;" :value="100"></ProgressBar>
          <img v-if="generating" class="ImageDiv" :src="image" alt="Dynamic Base64 Image">
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import hljs from 'highlight.js';
import CodeEditor from "simple-code-editor";
import ProgressBar from 'primevue/progressbar';
import Dropdown from 'primevue/dropdown';


export default {
  components: {
    CodeEditor,
    ProgressBar,
    Dropdown
  },
  name: 'Main',
  data () {
    return {
		    btnNum: 1,
        code: "",
        image: '',
        step: 0,
        generating: false,
        start: false,
        finish: false,
        selectedModel: 0,
        models: [
          {name: "Classification Model", value: 0},
          {name: "N to N Model", value: 1}
        ]

			}
  },
  methods:{
		ChangeType(index) {
			this.btnNum = index;
		},
    GenerateImage() {
      this.socket = new WebSocket('ws://localhost:8000/ws/get_image/');
      this.start = true
      this.finish = false
      this.step = 0
      this.generating = false
      this.socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (typeof data.image !== 'undefined'){
          if (!this.generating){
            this.generating = true
          }
          this.image = "data:image/png;base64," + data.image
        }
        this.step = data.step

        if (data.step === 200){
          this.socket.close()
          this.finish = true
        }
      };

      this.socket.onclose = (event) => {
        console.log('WebSocket closed:', event);
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    },
    getContent(content) {
      return content
    }
	}
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

.main-window {
  width: 99%;
  height: 99%;
  margin: auto;
  overflow: hidden;
}

.title {
  width: 100%;
  height: 10%;
  display: flex;
  flex-direction: column;
  background-image: url('../assets/nus_logo.png');
  background-size: 165px 75px;
  background-position: left;
  background-repeat: no-repeat;
}


#title-text {
  width: 100%;
  height: 50%;
  font-size: 40px;
  color: white;
  font-family: "Cambria";
  font-weight: bolder;
}


#author-text {
  width: 100%;
  height: 50%;
  color: white;
  margin-right: 10%;
  font-family: "Comic Sans";
}

.generating{
  width: 80%;
  height: 60%;
  margin: auto;
  margin-top: 5%
}

.ImageDiv {
  width: 50%;
  height: 50%;
  margin: auto;
  object-fit: contain;
}

.p-dropdown {
  background: black;
  color: white;
  font-weight: bold;
  border: 2px solid gray;
}

.p-dropdown-panel .p-dropdown-items {
    padding: 0.5rem 0;
    background: black;
}



.function-area {
  width: 100%;
  height: 85%;
  margin: auto;
  margin-top: 3%;
  display: flex;
}

.input-window {
  width: 45%;
  height: 100%;
  margin: auto;
}

#button-area {
  width: 100%;
  height: 10%;
}

.type-button {
  margin-left: 5%;
  margin-top: 1%;
  width: 12%;
  height: 60%;
  float: left;
  font-size: 18px;
  font-weight: 400;
}

.active {
  background: #7c94ff !important;
}

#editor {
  width: 100%;
  height: 77%;
}

#selector {
  width: 100%;
  height: 7%;
  
}

#generate-button{
  width: 15%;
  height: 60%;
  border: 0px;
  border-radius: 5px;
  font-size: 25px;
  font-weight: 200;
  margin-top: 1.5%;
  float: right;
  font-family: "Cambria";
  color: white;
  background-color: #7c94ff;
}

.generate_text{
  font-size: 150%;
  font-weight: bold;
  margin-bottom: 1%;
  font-family: trebuchet ms;
  color: white;
}

#generate-button:hover{
  width: 15%;
  height: 60%;
  border: 0px;
  font-size: 25px;
  font-weight: bolder;
  border-radius: 5px;
  margin-top: 1.5%;
  float: right;
  font-family: "Cambria";
  color: white;
  background-color: #3f63ff;
}

.output-window {
  border-radius: 5px;
  width: 45%;
  height: 92%;
  margin: auto;
  margin-top: 0;
  border: 2px dashed gray;
}

</style>


<style>
.p-highlight {
    color: #495057;
    background: rgb(73, 73, 73) !important;
}

.p-dropdown-item:hover{
  background: gray !important;
}
</style>