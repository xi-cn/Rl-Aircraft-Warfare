# main.py
import pygame
import sys
import traceback
import myplane
import enemy
import bullet
import supply

from pygame.locals import *
from random import *

class Maze:
    
    def __init__(self) -> None:
        pygame.init()
        # pygame.mixer.init()

        self.bg_size = self.width, self.height = 480, 800
        self.screen = pygame.display.set_mode(self.bg_size)
        pygame.display.set_caption("飞机大战 -- FishC Demo")

        self.background = pygame.image.load("images/background_black.png").convert()

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)

        # 得分标志
        self.score_font = pygame.font.Font("font/font.ttf", 36)
        
        # 标志是否暂停游戏
        self.paused = False
        self.pause_nor_image = pygame.image.load("images/pause_nor.png").convert_alpha()
        self.pause_pressed_image = pygame.image.load("images/pause_pressed.png").convert_alpha()
        self.resume_nor_image = pygame.image.load("images/resume_nor.png").convert_alpha()
        self.resume_pressed_image = pygame.image.load("images/resume_pressed.png").convert_alpha()

        # 炸弹
        self.bomb_image = pygame.image.load("images/bomb.png").convert_alpha()
        self.bomb_font = pygame.font.Font("font/font.ttf", 48)

        # 生命
        self.life_image = pygame.image.load("images/life.png").convert_alpha()

        # 游戏结束
        self.gameover_font = pygame.font.Font("font/font.ttf", 48)
        self.again_image = pygame.image.load("images/again.png").convert_alpha()
        self.gameover_image = pygame.image.load("images/gameover.png").convert_alpha()

    def add_small_enemies(self, group1, group2, num):
        for i in range(num):
            e1 = enemy.SmallEnemy(self.bg_size)
            group1.add(e1)
            group2.add(e1)


    def add_mid_enemies(self, group1, group2, num):
        for i in range(num):
            e2 = enemy.MidEnemy(self.bg_size)
            group1.add(e2)
            group2.add(e2)


    def add_big_enemies(self, group1, group2, num):
        for i in range(num):
            e3 = enemy.BigEnemy(self.bg_size)
            group1.add(e3)
            group2.add(e3)


    def inc_speed(self, target, inc):
        for each in target:
            each.speed += inc

    def reset(self):
        
        # 生成我方飞机
        self.me = myplane.MyPlane(self.bg_size)
        
        self.enemies = pygame.sprite.Group()
        
        # 生成敌方小型飞机
        self.small_enemies = pygame.sprite.Group()
        self.add_small_enemies(self.small_enemies, self.enemies, 30)
        
        # 生成敌方中型飞机
        self.mid_enemies = pygame.sprite.Group()
        self.add_mid_enemies(self.mid_enemies, self.enemies, 15)
        
        # 生成敌方大型飞机
        self.big_enemies = pygame.sprite.Group()
        self.add_big_enemies(self.big_enemies, self.enemies, 10)
        # 生成普通子弹
        self.bullet1 = []
        self.bullet1_index = 0
        self.BULLET1_NUM = 5
        for i in range(self.BULLET1_NUM):
            self.bullet1.append(bullet.Bullet1(self.me.rect.midtop))
        
        # 生成超级子弹
        self.bullet2 = []
        self.bullet2_index = 0
        self.BULLET2_NUM = 10
        for i in range(self.BULLET2_NUM // 2):
            self.bullet2.append(bullet.Bullet2((self.me.rect.centerx - 33, self.me.rect.centery)))
            self.bullet2.append(bullet.Bullet2((self.me.rect.centerx + 30, self.me.rect.centery)))
        
        self.clock = pygame.time.Clock()
        
        # 中弹图片索引
        self.e1_destroy_index = 0
        self.e2_destroy_index = 0
        self.e3_destroy_index = 0
        self.me_destroy_index = 0
        
        # 统计得分
        self.score = 0
        # 暂停
        self.paused_rect = self.pause_nor_image.get_rect()
        self.paused_rect.left, self.paused_rect.top = self.width - self.paused_rect.width - 10, 10
        self.paused_image = self.pause_nor_image
        
        # 设置难度级别
        self.level = 1
        
        # 全屏炸弹
        self.bomb_rect = self.bomb_image.get_rect()
        self.bomb_num = 3
        
        # 每30秒发放一个补给包
        self.bullet_supply = supply.Bullet_Supply(self.bg_size)
        self.bomb_supply = supply.Bomb_Supply(self.bg_size)
        self.SUPPLY_TIME = USEREVENT
        pygame.time.set_timer(self.SUPPLY_TIME, 5 * 1000)
        
        # 超级子弹定时器
        self.DOUBLE_BULLET_TIME = USEREVENT + 1
        
        # 标志是否使用超级子弹
        self.is_double_bullet = False
        
        # 解除我方无敌状态定时器
        self.INVINCIBLE_TIME = USEREVENT + 2
        
        # 生命数量
        self.life_rect = self.life_image.get_rect()
        self.life_num = 1
        
        # 用于阻止重复打开记录文件
        self.recorded = False
        
        # 游戏结束画面
        self.again_rect = self.again_image.get_rect()
        self.gameover_rect = self.gameover_image.get_rect()
        
        # 用于切换图片
        self.switch_image = True
        
        # 用于延迟
        self.delay = 100
        
        self.running = True

        # 游戏图像
        current_surface = pygame.display.get_surface()  
        # RGB格式
        observation = pygame.surfarray.array3d(current_surface)  
        # 转化为 c w h
        observation = observation.transpose((2, 0 , 1))

        return self.step(2)


    def step(self, action):
        
        # 奖励反馈
        reward = 0
        # 游戏结束
        done = False

        for iter in range(self.epochs):
            # 特殊事件
            for event in pygame.event.get():  
                
                if event.type == self.SUPPLY_TIME:
                    # self.supply_sound.play()
                    if choice([True, False]):
                        self.bomb_supply.reset()
                    else:
                        self.bullet_supply.reset()
                
                elif event.type == self.DOUBLE_BULLET_TIME:
                    self.is_double_bullet = False
                    pygame.time.set_timer(self.DOUBLE_BULLET_TIME, 0)
                
                elif event.type == self.INVINCIBLE_TIME:
                    self.me.invincible = False
                    pygame.time.set_timer(self.INVINCIBLE_TIME, 0)

            

            # 根据用户的得分增加难度
            if self.level == 1 and self.score > 50000:
                self.level = 2
                # self.upgrade_sound.play()
                # 增加3架小型敌机、2架中型敌机和1架大型敌机
                self.add_small_enemies(self.small_enemies, self.enemies, 15)
                self.add_mid_enemies(self.mid_enemies, self.enemies, 10)
                self.add_big_enemies(self.big_enemies, self.enemies, 5)
                # 提升小型敌机的速度
                self.inc_speed(self.small_enemies, 1)
            elif self.level == 2 and self.score > 300000:
                self.level = 3
                # self.upgrade_sound.play()
                # 增加15架小型敌机、10架中型敌机和5架大型敌机
                self.add_small_enemies(self.small_enemies, self.enemies, 15)
                self.add_mid_enemies(self.mid_enemies, self.enemies, 10)
                self.add_big_enemies(self.big_enemies, self.enemies, 5)
                # 提升小型敌机的速度
                self.inc_speed(self.small_enemies, 1)
                self.inc_speed(self.mid_enemies, 1)
            elif self.level == 3 and self.score > 600000:
                self.level = 4
                # self.upgrade_sound.play()
                # 增加15架小型敌机、10架中型敌机和5架大型敌机
                self.add_small_enemies(self.small_enemies, self.enemies, 15)
                self.add_mid_enemies(self.mid_enemies, self.enemies, 10)
                self.add_big_enemies(self.big_enemies, self.enemies, 5)
                # 提升小型敌机的速度
                self.inc_speed(self.small_enemies, 1)
                self.inc_speed(self.mid_enemies, 1)
            elif self.level == 4 and self.score > 1000000:
                self.level = 5
                # self.upgrade_sound.play()
                # 增加15架小型敌机、10架中型敌机和5架大型敌机
                self.add_small_enemies(self.small_enemies, self.enemies, 15)
                self.add_mid_enemies(self.mid_enemies, self.enemies, 10)
                self.add_big_enemies(self.big_enemies, self.enemies, 5)
                # 提升小型敌机的速度
                self.inc_speed(self.small_enemies, 1)
                self.inc_speed(self.mid_enemies, 1)
            
            self.screen.blit(self.background, (0, 0))
            
            if self.life_num and not self.paused:

                if action == 0:
                    self.me.moveLeft()
                elif action == 1:
                    self.me.moveRight()
                elif action == 2:
                    self.me.moveUp()
                elif action == 3:
                    self.me.moveDown()
                
                # 绘制全屏炸弹补给并检测是否获得
                if self.bomb_supply.active:
                    self.bomb_supply.move()
                    self.screen.blit(self.bomb_supply.image, self.bomb_supply.rect)
                    if pygame.sprite.collide_mask(self.bomb_supply, self.me):
                        # self.get_bomb_sound.play()
                        if self.bomb_num < 3:
                            self.bomb_num += 1
                        self.bomb_supply.active = False
                
                # 绘制超级子弹补给并检测是否获得
                if self.bullet_supply.active:
                    self.bullet_supply.move()
                    self.screen.blit(self.bullet_supply.image, self.bullet_supply.rect)
                    if pygame.sprite.collide_mask(self.bullet_supply, self.me):
                        # self.get_bullet_sound.play()
                        self.is_double_bullet = True
                        pygame.time.set_timer(self.DOUBLE_BULLET_TIME, 18 * 1000)
                        self.bullet_supply.active = False
                        reward += self.reward[4]
                
                # 发射子弹
                if not (self.delay % 10):
                    # self.bullet_sound.play()
                    if self.is_double_bullet:
                        self.bullets = self.bullet2
                        self.bullets[self.bullet2_index].reset((self.me.rect.centerx - 33, self.me.rect.centery))
                        self.bullets[self.bullet2_index + 1].reset((self.me.rect.centerx + 30, self.me.rect.centery))
                        self.bullet2_index = (self.bullet2_index + 2) % self.BULLET2_NUM
                    else:
                        self.bullets = self.bullet1
                        self.bullets[self.bullet1_index].reset(self.me.rect.midtop)
                        self.bullet1_index = (self.bullet1_index + 1) % self.BULLET1_NUM
                
                # 检测子弹是否击中敌机
                for b in self.bullets:
                    if b.active:
                        b.move()
                        self.screen.blit(b.image, b.rect)
                        enemy_hit = pygame.sprite.spritecollide(b, self.enemies, False, pygame.sprite.collide_mask)
                        if enemy_hit:
                            b.active = False
                            for e in enemy_hit:
                                if e in self.mid_enemies or e in self.big_enemies:
                                    e.hit = True
                                    e.energy -= 1
                                    if e.energy == 0:
                                        e.active = False
                                else:
                                    e.active = False
                
                # 绘制大型敌机
                for each in self.big_enemies:
                    if each.active:
                        each.move()
                        if each.hit:
                            self.screen.blit(each.image_hit, each.rect)
                            each.hit = False
                        else:
                            if self.switch_image:
                                self.screen.blit(each.image1, each.rect)
                            else:
                                self.screen.blit(each.image2, each.rect)
                        
                        # 绘制血槽
                        pygame.draw.line(self.screen, self.BLACK, \
                                        (each.rect.left, each.rect.top - 5), \
                                        (each.rect.right, each.rect.top - 5), \
                                        2)
                        # 当生命大于20%显示绿色，否则显示红色
                        energy_remain = each.energy / enemy.BigEnemy.energy
                        if energy_remain > 0.2:
                            energy_color = self.GREEN
                        else:
                            energy_color = self.RED
                        pygame.draw.line(self.screen, energy_color, \
                                        (each.rect.left, each.rect.top - 5), \
                                        (each.rect.left + each.rect.width * energy_remain, \
                                        each.rect.top - 5), 2)
                        
                        # 即将出现在画面中，播放音效
                        if each.rect.bottom == -50:
                            pass
                            # self.enemy3_fly_sound.play(-1)
                    else:
                        # 毁灭
                        if not (self.delay % 3):
                            if self.e3_destroy_index == 0:
                                reward += self.reward[3]
                                # self.enemy3_down_sound.play()
                            self.screen.blit(each.destroy_images[self.e3_destroy_index], each.rect)
                            self.e3_destroy_index = (self.e3_destroy_index + 1) % 6
                            if self.e3_destroy_index == 0:
                                # self.enemy3_fly_sound.stop()
                                self.score += 10000
                                each.reset()

                
                # 绘制中型敌机：
                for each in self.mid_enemies:
                    if each.active:
                        each.move()
                        
                        if each.hit:
                            self.screen.blit(each.image_hit, each.rect)
                            each.hit = False
                        else:
                            self.screen.blit(each.image, each.rect)
                        
                        # 绘制血槽
                        pygame.draw.line(self.screen, self.BLACK, \
                                        (each.rect.left, each.rect.top - 5), \
                                        (each.rect.right, each.rect.top - 5), \
                                        2)
                        # 当生命大于20%显示绿色，否则显示红色
                        energy_remain = each.energy / enemy.MidEnemy.energy
                        if energy_remain > 0.2:
                            energy_color = self.GREEN
                        else:
                            energy_color = self.RED
                        pygame.draw.line(self.screen, energy_color, \
                                        (each.rect.left, each.rect.top - 5), \
                                        (each.rect.left + each.rect.width * energy_remain, \
                                        each.rect.top - 5), 2)
                    else:
                        # 毁灭
                        if not (self.delay % 3):
                            if self.e2_destroy_index == 0:
                                reward += self.reward[2]
                                # self.enemy2_down_sound.play()
                            self.screen.blit(each.destroy_images[self.e2_destroy_index], each.rect)
                            self.e2_destroy_index = (self.e2_destroy_index + 1) % 4
                            if self.e2_destroy_index == 0:
                                self.score += 6000
                                each.reset()
                
                # 绘制小型敌机：
                for each in self.small_enemies:
                    if each.active:
                        each.move()
                        self.screen.blit(each.image, each.rect)
                    else:
                        # 毁灭
                        if not (self.delay % 3):
                            if self.e1_destroy_index == 0:
                                reward += self.reward[1]
                                # self.enemy1_down_sound.play()
                            self.screen.blit(each.destroy_images[self.e1_destroy_index], each.rect)
                            self.e1_destroy_index = (self.e1_destroy_index + 1) % 4
                            if self.e1_destroy_index == 0:
                                self.score += 1000
                                each.reset()

                
                # 检测我方飞机是否被撞
                enemies_down = pygame.sprite.spritecollide(self.me, self.enemies, False, pygame.sprite.collide_mask)
                if enemies_down and not self.me.invincible:
                    self.me.active = False
                    for e in enemies_down:
                        e.active = False
                
                # 绘制我方飞机
                if self.me.active:
                    if self.switch_image:
                        self.screen.blit(self.me.image1, self.me.rect)
                    else:
                        self.screen.blit(self.me.image2, self.me.rect)
                else:
                    # 毁灭
                    if not (self.delay % 1):
                        if self.me_destroy_index == 0:
                            reward = self.reward[0]
                            done = True
                            # self.me_down_sound.play()
                        self.screen.blit(self.me.destroy_images[self.me_destroy_index], self.me.rect)
                        self.me_destroy_index = (self.me_destroy_index + 1) % 4
                        if self.me_destroy_index == 0:
                            self.life_num -= 1
                            self.me.reset()
                            pygame.time.set_timer(self.INVINCIBLE_TIME, 3 * 1000)
                
                # 绘制全屏炸弹数量
                bomb_text = self.bomb_font.render("× %d" % self.bomb_num, True, self.WHITE)
                text_rect = bomb_text.get_rect()
                self.screen.blit(self.bomb_image, (10, self.height - 10 - self.bomb_rect.height))
                self.screen.blit(bomb_text, (20 + self.bomb_rect.width, self.height - 5 - text_rect.height))
                
                # 绘制剩余生命数量
                if self.life_num:
                    for i in range(self.life_num):
                        self.screen.blit(self.life_image, \
                                    (self.width - 10 - (i + 1) * self.life_rect.width, \
                                    self.height - 10 - self.life_rect.height))
                
                # 绘制得分
                score_text = self.score_font.render("Score : %s" % str(self.score), True, self.WHITE)
                self.screen.blit(score_text, (10, 5))
            
            if done:
                break


            self.screen.blit(self.paused_image, self.paused_rect)
            
            # 切换图片
            if not (self.delay % 5):
                self.switch_image = not self.switch_image
            
            self.delay -= 1
            if not self.delay:
                self.delay = 100
            
            pygame.display.flip()
            self.clock.tick(self.tick)

        # 游戏图像
        current_surface = pygame.display.get_surface()  
        # RGB格式
        observation = pygame.surfarray.array3d(current_surface)

        score = self.score
        
        if reward == 0:
            reward = self.reward[5]
        
        return observation, reward, done, score
    
    def distory(self):
        pygame.quit()
        # sys.exit()

    def pause(self):
        pygame.time.set_timer(self.SUPPLY_TIME, 0)
        # pygame.mixer.music.pause()
        # pygame.mixer.pause()
    def proceed(self):
        pygame.time.set_timer(self.SUPPLY_TIME, 30 * 1000)
        # pygame.mixer.music.unpause()
        # pygame.mixer.unpause()

    # 设置一次迭代循环次数
    def setStepBatchs(self, epochs=10):
        self.epochs = epochs

    # 设置帧数
    def setTick(self, tick=60):
        self.tick = tick
    
    # 设置奖励
    def setReward(self, reward):
        self.reward = reward

if __name__ == "__main__":
    maze = Maze()
    maze.reset()
    while True:
        maze.step(0)