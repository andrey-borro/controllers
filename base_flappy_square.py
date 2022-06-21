import pygame
import pygame.freetype
import random

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 650

COLOUR_WHITE = (255, 255, 255)
COLOUR_BLACK = (0,0,0)
COLOUR_BACKGROUND = (145, 220, 250)
COLOUR_SQUARE = (240,150,30)
COLOUR_PIPE = (35,215,80)
COLOUR_BORDER = (255,215,160)

SQUARE_WIDTH, SQUARE_HEIGHT = 54, 49
SQUARE_HITBOX_MARGIN = 4
SQUARE_X, SQUARE_Y = 125, 300
SQUARE_DOWN_ACCELERATION = 0.5
SQUARE_JUMP_SPEED = -8.25

PIPE_WIDTH = 100
PIPE_GAP_SIZE = 150
PIPE_DISTANCE = 350
PIPE_COUNT = 3
PIPE_SPEED = -2
PIPE_BOUNDS_TOP = int(0.2 * SCREEN_HEIGHT)
PIPE_BOUNDS_BOTTOM = int(0.8 * SCREEN_HEIGHT)

BORDER_HEIGHT = 25

DELETION_MARGIN = 10

class Pipe():
    def __init__(self, x: int, count: int) -> None: 
        self.x = x
        self.y = random.randrange(PIPE_BOUNDS_TOP, PIPE_BOUNDS_BOTTOM)
        self.count = count
        self.rects = [
            pygame.Rect(self.x, 0, PIPE_WIDTH, self.y - PIPE_GAP_SIZE//2),
            pygame.Rect(self.x, self.y + PIPE_GAP_SIZE//2, PIPE_WIDTH,SCREEN_HEIGHT - self.y - PIPE_GAP_SIZE//2)
        ]

    def draw(self, screen: pygame.Surface, font: pygame.freetype.Font) -> None:
        for rect in self.rects:
            pygame.draw.rect(screen, COLOUR_PIPE, rect)
        
        text_surface, text_rect = font.render(str(self.count))
        screen.blit(text_surface, (self.x + (PIPE_WIDTH - text_rect.width) // 2, 
                                    self.y - PIPE_GAP_SIZE //2 - 15 - text_rect.height))
    
    def move(self) -> None:
        self.rects = [rect.move(PIPE_SPEED,0) for rect in self.rects]
        self.x += PIPE_SPEED
    
def game_loop() -> int:
    pygame.init()
    pygame.freetype.init()
    pygame.display.set_caption('Flappy Square')
    pygame.display.set_icon(pygame.image.load('icon.png'))

    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont('Arial', 23)
    main_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    score = 0
    game_over = False

    square = {
            'speed' : 0, 
            'body': pygame.Rect(SQUARE_X, SQUARE_Y, SQUARE_WIDTH, SQUARE_HEIGHT),
            'hitbox': pygame.Rect(SQUARE_X + SQUARE_HITBOX_MARGIN//2, SQUARE_Y + SQUARE_HITBOX_MARGIN//2,
                                    SQUARE_WIDTH - SQUARE_HITBOX_MARGIN, SQUARE_HEIGHT - SQUARE_HITBOX_MARGIN)
    }
    
    borders = [pygame.Rect(0, y,SCREEN_WIDTH, BORDER_HEIGHT) for y in [0, SCREEN_HEIGHT - BORDER_HEIGHT]]
    pipes = [Pipe(int(SCREEN_WIDTH * 1.5) + i * PIPE_DISTANCE, i + 1) for i in range(PIPE_COUNT)]

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: 
                square['speed'] = SQUARE_JUMP_SPEED
        square['speed'] += SQUARE_DOWN_ACCELERATION
        square['body'], square['hitbox'] = (x.move(0, square['speed']) for x in (square['body'], square['hitbox']))

        main_screen.fill(COLOUR_BACKGROUND)
        pygame.draw.rect(main_screen, COLOUR_SQUARE, square['body'])

        for index in range(len(pipes) - 1, -1, -1):
            pipe = pipes[index]
            pipe.move()
            if pipe.x + PIPE_WIDTH + DELETION_MARGIN < 0:
                pipes.pop(index)
                pipes.append(Pipe(pipes[-1].x + PIPE_DISTANCE, pipes[-1].count + 1))
            pipe.draw(main_screen, font)
            
        for border in borders:
            pygame.draw.rect(main_screen, COLOUR_BORDER, border)
        
        game_over = any(square['hitbox'].colliderect(x) for x in borders + sum([pipe.rects for pipe in pipes], []))
        score += 1.0/60.0 #not quite how score works in flappy bird but I like this more
        font.render_to(main_screen, (5,5), f'Score: {score:.2f}', COLOUR_BLACK)
        pygame.display.update()
        clock.tick(60)
    return score

def launch_game() -> None:
    #game start function
    score = game_loop()
    print(f'You got a score of {score:.2f}')


