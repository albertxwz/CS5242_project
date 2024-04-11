<template>
  <div id="app" style="text-align: center;">
    <Main></Main>
  </div>
</template>

<script>
import Main from "./components/main.vue"
export default {
  components: {
    Main
  },
  name: 'App'
}
</script>

<style>
html, body, #app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  /* color: #2c3e50; */
  width: 100%;
  height: 100%;
  padding: 0%;
  margin: 0%;
  background-image: url('assets/background.jpg');
}
</style>
