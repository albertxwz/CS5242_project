// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'

import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

import PrimeVue from 'primevue/config';


Vue.use(ElementUI)
Vue.use(PrimeVue);

import 'primevue/resources/themes/saga-blue/theme.css'; // 主题
import 'primevue/resources/primevue.min.css'; // 核心样式
import 'primeicons/primeicons.css'; // 图标

Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
