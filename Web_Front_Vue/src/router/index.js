import Vue from 'vue'
import VueRouter from 'vue-router'
import Test1 from "@/components/Test1";
import Test2 from "@/components/Test2";
import TestArea from "@/components/TestArea";

Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'testArea',
    component: TestArea
  },
  {
    path: '/test1',
    name: 'test1',
    component: Test1
  },
  {
    path: '/test2',
    name: 'test2',
    component: Test2
  }
]

const router = new VueRouter({
  base: process.env.BASE_URL,
  routes
})

export default router
