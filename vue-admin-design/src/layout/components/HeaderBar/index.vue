<template>
  <div class="header-bar clear-fix">
    <FoldSideMenu />
    <BreadCrumbs />
    <div class="header-right clear-fix">
      <FullScreen />
      <UserAvatar />
    </div>
  </div>
</template>

<script>
import FoldSideMenu from '../../../components/FoldSideMenu'
import BreadCrumbs from '../../../components/BreadCrumbs'
import FullScreen from '../../../components/FullScreen'
import UserAvatar from '../../../components/UserAvatar'

export default {
  components: {
    FoldSideMenu,
    BreadCrumbs,
    FullScreen,
    UserAvatar
  }
}
</script>

<style lang="less">
  .header-bar {
    height: 32px;
    padding: 16px 20px;
    .header-right {
      float: right;
      width: 200px;
    }
  }
</style>
