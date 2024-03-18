<template>
  <div class="main-window">
    <div class="title">
      <span id="title-text">Markup to Image Demo</span>
      <span id="author-text"><i style="float: right; margin-right: 5%;">Zhang Xuan, Xie Wenzheng, Chen Feitong, Liu Zhicheng</i></span>
    </div>
    <div class="function-area">
      <div class="input-window">
        <div id="editor">
          <CodeEditor 
            :wrap="true"
            :modelValue.sync="code"
            @content="getContent"
            theme="github-dark" 
            :languages="[['latex', 'LaTex'],['html', 'HTML']]"
            width="100%" 
            height="650px">
          </CodeEditor>
        </div>

        <div style="width: 100%; height: 10%;">
          <button id="generate-button" @click="GenerateImage()">Generate</button>
        </div>
      </div>
      <div class="output-window"></div>
    </div>
  </div>
</template>

<script>
import hljs from 'highlight.js';
import CodeEditor from "simple-code-editor";
import axios from 'axios';
import { Loading } from 'element-ui';

export default {
  components: {
    CodeEditor
  },
  name: 'Main',
  data () {
    return {
		    btnNum: 1,
        code: "",
			}
  },
  methods:{
		ChangeType(index) {
			this.btnNum = index;
		},
    GenerateImage() {
      console.log(this.code)

      const url = 'http://localhost:8000/generateImage/';
  
      const data = {
        "code": this.code
      };
  
      const config = {
        headers: {
          'Content-Type': 'application/json'
        }
      };
      const loading = this.$loading({
          lock: true,
          text: 'Loading',
          spinner: 'el-icon-loading',
          background: 'rgba(0, 0, 0, 0.7)'
        });
      
      axios.post(url, data, config)
        .then(response => {
          console.log(response.data);
          loading.close();
          this.$notify({
            title: 'Success',
            message: 'The Image Has been Generated!',
            type: 'success'
          });
        })
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
  font-family: "Cambria";
  font-weight: bolder;
}


#author-text {
  width: 100%;
  height: 50%;
  margin-right: 10%;
  font-family: "Comic Sans";
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

.editor {
  width: 100%;
  height: 90%;
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
  background-color: #7c94ff;
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
  background-color: #3f63ff;
}

.output-window {
  border-radius: 5px;
  width: 45%;
  height: 92%;
  margin: auto;
  margin-top: 0;
  border: 2px dashed black;
}

</style>
